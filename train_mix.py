# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from model import DM2FNet, DM2FNet_new
from model_mix import MixDehazeNet_s
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset, ItsDataset_mix, SotsDataset_mix
from tools.utils import AvgMeter, check_mkdir
from torch.cuda.amp import autocast, GradScaler
from MixDehazeNet.utils.CR_res import ContrastLoss_res
from MixDehazeNet.datasets.loader import PairLoader_its, PairLoader_sots
from torch.cuda.amp import autocast, GradScaler

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

scaler = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train an improved DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS_mixDehazenet',
        help='experiment name.')
    args = parser.parse_args()

    return args


# cfgs = {
#     'use_physical': True,
#     'iter_num': 40000,
#     'train_batch_size': 16,
#     'last_iter': 0,
#     'lr': 5e-4,
#     'lr_decay': 0.9,
#     'weight_decay': 0,
#     'momentum': 0.9,
#     'snapshot': '',
#     'val_freq': 5000,
#     'crop_size': 256
# }

cfgs = {
    'use_physical': True,
    'iter_num': 4000,
    'train_batch_size': 32,
    'last_iter': 0,
    'lr': 2e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 400,
    'crop_size': 256
}

setting = {
    "batch_size": 36,
    "patch_size": 256,
    "valid_mode": "test",
    "edge_decay": 0,
    "only_h_flip": False,
    "optimizer": "adamw",
    "lr": 2e-4,
    "epochs":100,
    "eval_freq": 1
}


criterion = None

def main():
    net = MixDehazeNet_s().cuda().train()
    # net = nn.DataParallel(net)
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=setting['lr'])
    # optimizer = optim.Adam([
    #     {'params': [param for name, param in net.named_parameters()
    #                 if name[-4:] == 'bias' and param.requires_grad],
    #      'lr': 2 * cfgs['lr']},
    #     {'params': [param for name, param in net.named_parameters()
    #                 if name[-4:] != 'bias' and param.requires_grad],
    #      'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    # ])

    criterion = []
    criterion.append(nn.L1Loss())
    criterion.append(ContrastLoss_res(ablation=False).cuda())

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfgs['iter_num'], eta_min=cfgs['lr'] * 1e-2)
    scaler = GradScaler()

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, optimizer)


def train(net, optimizer):
    curr_iter = cfgs['last_iter']
    criterion = []
    criterion.append(nn.L1Loss())
    criterion.append(ContrastLoss_res(ablation=False).cuda())

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()
    iter = 0

    for epoch in tqdm(range(0,setting['epochs'] + 1)):
        # print(optimizer.param_groups[0]['lr'])
        train_loss_record = AvgMeter()

        for data in train_loader:
            haze = data['source']
            gt = data['target']

            batch_size = haze.size(0)

            haze = haze.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            output = net(haze)
            loss = criterion[0](output, gt)+criterion[1](output, gt, haze)*0.1

            # loss.backward()
            train_loss_record.update(loss.item(), batch_size)
            iter += 1
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # print('loss:', loss.item())
            log = '[iter %d], [loss %.5f]' % (
                iter, loss.item())
            print(log)
            open(log_path, 'a').write(log + '\n')
            

        log = '[epoch %d], [loss %.5f], [lr %.13f]' % (
            epoch, train_loss_record.avg, optimizer.param_groups[0]['lr'])
        print(log)
        open(log_path, 'a').write(log + '\n')


        scheduler.step()

		# train_ls.append(loss)
		# idx.append(epoch)

		# writer.add_scalar('train_loss', loss, epoch)

        if epoch % setting['eval_freq'] == 0:
            validate(net, epoch, optimizer)

			# writer.add_scalar('valid_psnr', avg_psnr, epoch)


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze = data['source']
            gt = data['target']

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[0]['lr'])
    print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, loss_record.avg))
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # cudnn.benchmark = True
    # torch.cuda.set_device(int(args.gpus))

    train_dataset = PairLoader_its(TRAIN_ITS_ROOT, 'train', 
								setting['patch_size'],
							    setting['edge_decay'],
							    setting['only_h_flip'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = PairLoader_sots(TEST_SOTS_ROOT, 'test', 'valid', 
							  setting['patch_size'])
    val_loader = DataLoader(val_dataset, batch_size=8)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()
