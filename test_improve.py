# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, TEST_HAZERD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy, DM2FNet_new
from model_improve import DM2FNet_woPhy_new, DM2FNet_attention_in, DM2FNet_new2, ours_wo_AFIM, DM2FNet_attention_chuan, DM2FNet_woPhy_attention_chuan, DM2FNet_woPhy_woAFIM
from model_improve2 import DM2FNet_plus_attention
from datasets import SotsDataset, OHazeDataset, HazeRDDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from pyciede2000 import ciede2000
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from model_mix import MixDehazeNet_s
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
# torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
# exp_name = 'RESIDE_ITS_improve'
# exp_name = 'O-Haze'
exp_name = 'O-Haze_improve'
# exp_name = 'RESIDE_ITS_DM2FNet_plus_attention'
# exp_name = 'RESIDE_ITS_DM2FNet_new2_2'
# exp_name = 'O-Haze_DM2FNet_woPhy_woAFIM'
# exp_name = 'RESIDE_ITS_DM2FNet_attention_chuan'
# exp_name = 'RESIDE_ITS_more_attention'
# exp_name = 'O-Haze_DM2FNet_woPhy_attention_chuan'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_19000_loss_0.04261_lr_0.000014',
    # 'snapshot': 'iter_40000_loss_0.01241_lr_0.000000',
    # 'snapshot': 'iter_20000_loss_0.04922_lr_0.000000',
    'snapshot': 'iter_20000_loss_0.04922_lr_0.000000',
    # 'snapshot': 'iter_1000_loss_0.07267_lr_0.000000',
    # 'snapshot': 'iter_40000_loss_0.01244_lr_0.000000',
    # 'snapshot': 'iter_20000_loss_0.04862_lr_0.000000',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    'O-Haze': OHAZE_ROOT,
    # 'HazeRD': TEST_HAZERD_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = ours_wo_AFIM().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy_new().cuda()
                dataset = OHazeDataset(root, 'test')
            elif 'HazeRD' in name:
                net = DM2FNet_attention_in().cuda()
                dataset = HazeRDDataset(root)
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, mses, ciede2000s = [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True, channel_axis = 2,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    ssims.append(ssim)
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim))
                    mse = mean_squared_error(gt, r)
                    mses.append(mse)
                    lab_r = rgb2lab(r)
                    lab_gt = rgb2lab(gt)
                    ciede2000 = deltaE_ciede2000(lab_gt, lab_r)
                    ciede2000s.append(ciede2000.mean())
                    print('predicting for {} ({}/{}) [{}]: MSE {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], mse))
                    print('predicting for {} ({}/{}) [{}]: CIEDE2000 {:.4f}'
                            .format(name, idx + 1, len(dataloader), fs[i], ciede2000.mean()))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))
                

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}")
            print(f"[{name}] MSE: {np.mean(mses):.6f}")
            print(f"[{name}] CIEDE2000: {np.mean(ciede2000s):.6f}")
            with open(os.path.join(ckpt_path, exp_name,'(%s) %s_%s' % (exp_name, name, args['snapshot']), 'result.txt'), 'w') as f:
                f.write(f"L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}\n")
                f.write(f"MSE: {np.mean(mses):.6f}\n")
                f.write(f"CIEDE2000: {np.mean(ciede2000s):.6f}\n")

            

if __name__ == '__main__':
    main()
