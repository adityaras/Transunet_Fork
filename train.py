import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default="../../input/co-skel-448x448/CoSkel/CoSkel+", help='root dir for data')
parser.add_argument('--load_checkpoint_path', type=str,
                    default=".", help='Path for sved model')
parser.add_argument('--dataset', type=str,
                    default='Co-Skel', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default="../../input/co-skel-448x448/CoSkel/CoSkel+/train.csv", help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=448, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--dice_flag', dest='dice_flag', action='store_true',
                    help='Using Dice loss')
parser.add_argument('--double_channel', dest='double_channel', action='store_true',
                    help='Using 2 channel')
parser.add_argument('--learn_from_pretrained', dest='learn_from_pretrained', action='store_true',
                    help='Using pretrained model')
parser.add_argument('--is_pretrain', dest='is_pretrain', action='store_true',
                    help='Continue training again')
parser.add_argument('--epochs_till_now', type=int,
                    default=0, help='epochs to start at if pretraining')
parser.add_argument('--model_path', type=str,
                    default="", help='path to pretrained model')
parser.add_argument('--selective-attention', dest='selective_attention', action='store_true',
                    help='Using selective attention')
parser.add_argument('--alpha_coeff', type=float,  default=1,
                    help='segmentation network learning rate')
parser.add_argument('--patch_mse_loss', action='store_true',
                    help='Using patch mse')
parser.add_argument('--gamma_coeff', type=float,  default=1,
                    help='MSE loss network learning rate')
args = parser.parse_args()


if __name__ == "__main__":
    #     print(args.dice_flag,args.double_channel)
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir'] 
    args.is_chkpnt = False
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + \
        str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path+'_' + \
        str(args.max_iterations)[
            0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + \
        str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + \
        str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s' + \
        str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.selective_attention = args.selective_attention
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(
            args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size,
                  num_classes=config_vit.n_classes).cuda()
    if args.learn_from_pretrained:
        net.load_from(weights=np.load(config_vit.pretrained_path))
        print(print("Loading pretrained model from path: ", config_vit.pretrained_path))
  
    if args.is_pretrain:
        net.load_state_dict(torch.load(args.model_path))
        print("Continuing to train Model from path: ", args.model_path)

    trainer = {args.dataset: trainer_synapse, }
    trainer[dataset_name](args, net, snapshot_path)
