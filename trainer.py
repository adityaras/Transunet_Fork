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

def BCELoss_class_weighted():

    def _one_hot_encoder(input_tensor):
        tensor_list = []
        for i in range(2):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def loss(inpt, target,weights,dc):
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
            bce = - weights[:,0,:,:] * target[:,1,:,:] * torch.log(inpt[:,1,:,:]) - (target[:,0,:,:]) * weights[:,1,:,:] * torch.log(inpt[:,0,:,:])
            return torch.mean(bce)
    return loss

def trainer_synapse(args, model, snapshot_path):
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = LoadData(args.list_dir,args.root_path,args.double_channel)
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
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch,weights,_ = sampled_batch[0], sampled_batch[1],sampled_batch[2],sampled_batch[3]
            image_batch, label_batch,weights = image_batch.cuda(), label_batch.cuda(),weights.cuda()
            
            outputs = model(image_batch)
#             print(image_batch.shape, outputs.shape,label_batch.shape)
#             print(outputs.shape,label_batch[:].long().shape,weights,label_batch.shape)
#             print(weights.shape)
#             exit()
            
            if args.dice_flag:
                label_batch = label_batch.squeeze()
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
#                 print(loss_dice)
                loss_ce = ce_loss(outputs, label_batch.long(),weights,args.double_channel)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
            else:
                loss_ce = ce_loss(outputs.squeeze(1), label_batch.squeeze(1)[:].long(),weights,args.double_channel)
                loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

        

#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)
            print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
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
