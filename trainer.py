import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from dataloader import LoadData
from prettytable import PrettyTable
import csv

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def BCELoss_class_weighted():

    def _one_hot_encoder(input_tensor):
        tensor_list = []
        for i in range(2):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
            
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def loss(inpt, target,weights,dc):
        # print(target.shape,inpt.shape)
        inpt = torch.softmax(inpt, dim=1)
        # print(target.shape,inpt.shape)
        if not dc:
            inpt = torch.clamp(inpt,min=1e-7,max=1-1e-7)
            inpt = inpt.squeeze()
            target = target.squeeze()

            # print(inpt.shape,target.shape,weights[:,0].shape)
            weights = torch.unsqueeze(weights,axis=2)
            weights = torch.unsqueeze(weights,axis=3)
            weights = torch.tile(weights,(1,1,inpt.shape[-2],inpt.shape[-1]))
            # print(weights[:,0)
            bce = - weights[:,0,:,:] * target * torch.log(inpt) - (1 - target) * weights[:,1,:,:] * torch.log(1 - inpt)
            return torch.mean(bce)
        else:
            inpt = torch.clamp(inpt,min=1e-7,max=1-1e-7)
            target = _one_hot_encoder(target)
            weights = torch.unsqueeze(weights,axis=2)
            weights = torch.unsqueeze(weights,axis=3)
            weights = torch.tile(weights,(1,1,inpt.shape[-2],inpt.shape[-1]))
            # print("inside BCE: ",inpt[:,1,:,:])
            bce = - weights[:,0,:,:] * target[:,1,:,:] * torch.log(inpt[:,1,:,:]) - (target[:,0,:,:]) * weights[:,1,:,:] * torch.log(inpt[:,0,:,:])
            return torch.mean(bce)
    return loss

class SkelRecallLoss():
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(2):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
            
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
        
    def loss(self, output, target, mse=False):
        target = self._one_hot_encoder(target)
        output = torch.softmax(output, dim=1)
        print("SKEL RECALL LOSS: ", target.shape, output.shape)

class Patch_MSE_Loss():
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(2):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
            
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
        
    def loss(self, output, target, mse=False):
        if mse:
            mseLoss = nn.MSELoss()
        target = self._one_hot_encoder(target)
        # print(target.shape,output.shape)
        output = torch.softmax(output, dim=1)
        # print(target.shape,output.shape)
        base_shape = target.shape
        height = base_shape[-2]
        width = base_shape[-1]
        loss = 0
        h_list = [0, height//2, height]
        w_list = [0, width//2, width]
        for h in range(2):
            for w in range(2):
                out_patch = output[:, :, h_list[h] : h_list[h+1], w_list[w] : w_list[w+1]]
                target_patch = target[:, :, h_list[h] : h_list[h+1], w_list[w] : w_list[w+1]]
                if mse:
                    loss += mseLoss(out_patch[:,1,:,:], target_patch[:,1,:,:])
                    print(loss)
                else:
                    # print(torch.sum(out_patch[:,1,:,:]) , torch.sum(target_patch[:,1,:,:]))
                    loss += torch.square(torch.sum(out_patch[:,1,:,:]) - torch.sum(target_patch[:,1,:,:]))
        return loss/4

def trainer_synapse(args, model, snapshot_path):
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    count_parameters(model)
    # max_iterations = args.max_iterations
    db_train = LoadData(args.list_dir,args.root_path, args.dilate_skel, args.double_channel)
#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#                                transform=transforms.Compose(
#                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
#     dice_flag = False
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = BCELoss_class_weighted()
#     print(ce_loss)
    if args.dice_flag:
        dice_loss = DiceLoss(num_classes)
    if args.patch_mse_loss:
        patch_mse_loss = Patch_MSE_Loss()
    if args.dilate_skel:
        skel_recall_loss = SkelRecallLoss()
        
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay= args.weight_decay)
    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay= args.weight_decay)
    if args.adamw:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay= args.weight_decay)

    writer = SummaryWriter(snapshot_path + '/log')
    path_for_log_writing = snapshot_path + '/log/loss_iter.log'
    iter_num = args.epochs_till_now * len(trainloader)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(args.epochs_till_now,max_epoch), ncols=70)
    loss_arr = []
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            if args.dilate_skel:
                image_batch, label_batch,weights, dilated_skel ,_ = sampled_batch[0], sampled_batch[1],sampled_batch[2],sampled_batch[3], sampled_batch[4]
                image_batch, label_batch,weights, dilated_skel = image_batch.cuda(), label_batch.cuda(),weights.cuda(), dilated_skel.cuda()
            else:
                image_batch, label_batch,weights,_ = sampled_batch[0], sampled_batch[1],sampled_batch[2],sampled_batch[3]
                image_batch, label_batch,weights = image_batch.cuda(), label_batch.cuda(),weights.cuda()                
            outputs = model(image_batch)
#             print(image_batch.shape, outputs.shape,label_batch.shape)
#             print(outputs.shape,label_batch[:].long().shape,weights,label_batch.shape)
#             print(weights.shape)
#             exit()
            loss = 0

            if args.dilate_skel:
                label_batch = label_batch.squeeze()
                loss_skell_recall = skel_recall_loss.loss(outputs, label_batch)
                loss += args.delta_coeff * loss_patch_mse
            if args.dice_flag:
                label_batch = label_batch.squeeze()
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
#                 print(loss_dice)
                loss_ce = ce_loss(outputs, label_batch.long(),weights,args.double_channel)
                loss += args.beta_coeff * loss_ce + args.alpha_coeff * loss_dice
            else:
                loss_ce = ce_loss(outputs.squeeze(1), label_batch.squeeze(1)[:].long(),weights,args.double_channel)
                loss += args.beta_coeff * loss_ce
            if args.patch_mse_loss:
                label_batch = label_batch.squeeze()
                loss_patch_mse = patch_mse_loss.loss(outputs, label_batch,args.patch_mse_dontcount)
                loss += args.gamma_coeff * loss_patch_mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', args.beta_coeff * loss_ce.item(), iter_num)
        

#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)
            if args.patch_mse_loss and args.dice_flag:
                writer.add_scalar('info/loss_patchmse', args.gamma_coeff * loss_patch_mse.item(), iter_num)
                writer.add_scalar('info/loss_dice', args.alpha_coeff * loss_dice.item(), iter_num)
                loss_arr.append([iter_num, loss.item(), args.beta_coeff * loss_ce.item(), args.alpha_coeff* loss_dice.item(), args.gamma_coeff * loss_patch_mse.item()])
                print('iteration %d : loss : %f, loss_ce: %f, weighted_loss_dice : %f, weighted_loss_patch_mse: %f' % (iter_num, loss.item(), args.beta_coeff * loss_ce.item(), args.alpha_coeff* loss_dice.item(), args.gamma_coeff * loss_patch_mse.item()))
            elif args.patch_mse_loss:
                writer.add_scalar('info/loss_patchmse', loss_patch_mse, iter_num)
                loss_arr.append([iter_num, loss.item(), args.beta_coeff * loss_ce.item(), 0, args.gamma_coeff * loss_patch_mse.item()])
                print('iteration %d : loss : %f, loss_ce: %f, weighted_loss_patch_mse: %f' % (iter_num, loss.item(), args.beta_coeff * loss_ce.item(), args.gamma_coeff * loss_patch_mse.item()))
            elif args.dice_flag:
                writer.add_scalar('info/loss_dice', loss_dice, iter_num)
                loss_arr.append([iter_num, loss.item(), args.beta_coeff * loss_ce.item(), args.alpha_coeff* loss_dice.item(), 0])
                print('iteration %d : loss : %f, loss_ce: %f, weighted_loss_dice: %f' % (iter_num, loss.item(), args.beta_coeff * loss_ce.item(), args.alpha_coeff* loss_dice.item()))
            else:
                loss_arr.append([iter_num, loss.item(), args.beta_coeff * loss_ce.item(), 0, 0])
                print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), args.beta_coeff * loss_ce.item()))
            with open(path_for_log_writing, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(loss_arr)
        save_interval = 2  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
