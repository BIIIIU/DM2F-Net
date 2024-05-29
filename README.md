# DM2F-Net-improved

By Yuqi Zhang

This repo is an improvement of
"[Deep Multi-Model Fusion for Single-Image Dehazing](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Deep_Multi-Model_Fusion_for_Single-Image_Dehazing_ICCV_2019_paper.pdf)"
(ICCV 2019), written by Zijun Deng at the South China University of Technology.
The original repo can be found at [here](hhttps://github.com/zijundeng/DM2F-Net/tree/master).

## Results

<div style="text-align: center;">
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-9wq8">Dataset</th>
    <th class="tg-9wq8" colspan="4">O-HAZE</th>
    <th class="tg-9wq8" colspan="4">HazeRD</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">Method</td>
    <td class="tg-9wq8">PSNR</td>
    <td class="tg-9wq8">SSIM</td>
    <td class="tg-9wq8">MSE</td>
    <td class="tg-9wq8">CIEDE</td>
    <td class="tg-9wq8">PSNR</td>
    <td class="tg-9wq8">SSIM</td>
    <td class="tg-9wq8">MSE</td>
    <td class="tg-9wq8">CIEDE</td>
  </tr>
  <tr>
    <td class="tg-9wq8">DM2F-Net</td>
    <td class="tg-9wq8">25.113 </td>
    <td class="tg-9wq8">0.7742 </td>
    <td class="tg-9wq8">0.0032 </td>
    <td class="tg-9wq8">5.2000 </td>
    <td class="tg-9wq8">14.212 </td>
    <td class="tg-9wq8">0.8145</td>
    <td class="tg-9wq8">0.0724 </td>
    <td class="tg-9wq8">16.8331 </td>
  </tr>
  <tr>
    <td class="tg-9wq8">DM2F-Net-improved</td>
    <td class="tg-9wq8">25.602 </td>
    <td class="tg-9wq8">0.7752 </td>
    <td class="tg-9wq8">0.0030</td>
    <td class="tg-9wq8">4.9714</td>
    <td class="tg-9wq8">15.774 </td>
    <td class="tg-9wq8">0.8291</td>
    <td class="tg-9wq8">0.0589 </td>
    <td class="tg-9wq8">15.5905 </td>
  </tr>
</tbody></table>
</div>

The checkpoint and dehazing results can be found at 
[Baidu Drive](https://pan.baidu.com/s/1Zh7siyQrjsRPuL0-MMSoVg?pwd=5fzu).

## Installation & Preparation

Make sure you have `Python>=3.7` installed on your machine.

**Environment setup:**

1. Create conda environment

       conda create -n dm2f
       conda activate dm2f

2. Install dependencies (test with PyTorch 2.3.0):

   1. Install pytorch==2.3.0 torchvision==0.18.0 

   2. Install other dependencies

          pip install -r requirements.txt

* Prepare the dataset

   * Download the RESIDE dataset from the [official webpage](https://sites.google.com/site/boyilics/website-builder/reside).

   * Download the O-Haze dataset from the [official webpage](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

   * Make a directory `./data` and create a symbolic link for uncompressed data, e.g., `./data/RESIDE`.

## Training

1. Set the path of datasets in tools/config.py
2. Run by ```python train.py```

Use pretrained ResNeXt (resnext101_32x8d) from torchvision.

*Hyper-parameters* of training were set at the top of *train.py*, and you can conveniently
change them as you need.

Training a model on a single NVIDIA A100 GPU takes about 4 hours.

## Testing

1. Set the path of five benchmark datasets in tools/config.py.
2. Put the trained model in `./ckpt/`.
2. Run by ```python test.py```

*Settings* of testing were set at the top of `test.py`, and you can conveniently
change them as you need.

## License

DM2F-Net-improved is released under the [MIT license](LICENSE).

## Citation

If you find the code helpful to your research, please give a star to me!

