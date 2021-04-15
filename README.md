# Image Steganography

## Introduction

Project for course *Network Security Theory and Practice(网络安全原理与实践)*.

This repo follows [HiDDeN](https://github.com/ando-khachatryan/HiDDeN).

> Pytorch implementation of paper "HiDDeN: Hiding Data With Deep  Networks" by Jiren Zhu*, Russell Kaplan*, Justin Johnson, and Li  Fei-Fei: https://arxiv.org/abs/1807.09937
>  *: These authors contributed equally
>
> The authors have Lua+Torch implementation here: https://github.com/jirenz/HiDDeN
>
> Note that this is a work in progress, and I was not yet able to fully reproduce the results of the original paper.

I reconstruct the code of the network, and implement other parts according to my requirements.  

I tried [RelGAN's](https://github.com/elvisyjlin/RelGAN-PyTorch) loss function here and observe a slightly better performance.

> A **PyTorch** implementation of [**RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes**](https://arxiv.org/abs/1908.07269)
>
> The paper is accepted to ICCV 2019. We also have the Keras version [here](https://github.com/willylulu/RelGAN-Keras).

I use *Celeba* to train the network, and keep the default parameters used in the original repo.

## Installation

- Clone this repo

  ```bash
  git clone https://github.com/tancik/StegaStamp.git
  cd ImageSteganograpy
  ```

- Python 3 required

- install PyTorch (tested with PyTorch 1.5.1)

- install TensorFlow (for the full function of tensorboard)

- Download dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Dataset

I use the *Celeba* *Align&Cropped* dataset.  You can download it from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and rename the dataset to `datasets/img_align_celeba`. Then resize the images to (64*64):

```bash
python data/ReformatFileStructure.py
```

The data directory has the following structure:

```
datasets/
    celeba_resize/
        train/
            images/
                train_image1.jpg
                train_image2.jpg
                ...
        test/
            images/
            	test_image1.jpg
            	test_image2.jpg
            	...
```

`train` and `test` folders are so that we can use the standard torchvision data loaders`torchvision.datasets.ImageFolder` without change.

## Running

### Train

```bash
python train.py
```

- You can check the default value of all parameters in the code or change these values in the command line as you wish.

- --gpu_id: the id of your gpu.

- --use_noise: use noises after encoding, you can check or change the default noise combination in the code. True for using noise. 

- --relative_loss: use RelGAN's loss function or not. True for use and False means use the same loss function as the paper's author.

- e.g.  This means training the models with RelGAN's loss and noises using GPU 3. 

  ```bash
  python train.py --use_noise True --relative_loss True --gpu_id 3
  ```

### Test

```bash
python test.py
```

- You can check the default value of all parameters in the code or change these values in the command line as you wish.

- --gpu_id: the id of your gpu.

- --use_noise: use noises after encoding, you can check or change the default noise combination in the code. True for using noise. 

- **note**: you must modify yout log file's name in the function `test(args, logger_name)`, the models corresponding to the logger_name would be tested.

- e.g.  This means testing the models with noises using GPU 3. 

  ```bash
  python test.py --use_noise True --gpu_id 3
  ```

### Tensorboard

To visualize the training and testing, run the following command and navigate to http://localhost:6006 in your browser. (You can choose your own port)

```bash
tensorboard --logdir=logs/ --port=6006
```

### Encoding and decoding a message

To encode or decode a single message into a single picture

```bash
python encoder_or_decoder.py
```

## Experiments

Since the time left for our project is limited, I only conducted the following four experiments. I may do futher research in the future.

| EXP_NAME       | train_err_ratio | test_erro_ratio |
| -------------- | --------------- | --------------- |
| original       | 0               | 0               |
| Rel loss       | 0               | 0               |
| noise          | 0.20            | 0.003           |
| Rel loss+noise | 0.16            | 0.003           |

`original:` trains the model using original loss function and without noises.

`Rel loss:` trains the model using RelGAN's loss function and without noises.

 `noise:` trains the model using origial loss function and with noises.

`Rel loss+noise`: trains the model using RelGAN's loss function and with noises.

`train_err_ratio:` the error ratio between decode info and original info during training.

`test_err_ratio:`  the error ratio between decode info and original info during testing.

`noises:`  the noises combination follows the original repo.

`the noise layer parameters`

>Noise Layer paremeters
>
> - *Crop((height_min,height_max),(width_min,width_max))*, where ***(height_min,height_max)\*** is a range from which we draw a random number and keep that fraction of the height of the original image. ***(width_min,width_max)\*** controls the same for the width of the image. Put it another way, given an image with dimensions ***H x W,\*** the Crop() will randomly crop this into dimensions ***H' x W'\***, where ***H'/H\*** is in the range ***(height_min,height_max)\***, and ***W'/W\*** is in the range ***(width_min,width_max)\***. In the paper, the authors use a single parameter ***p\*** which shows the ratio ***(H' \* W')/ (H \* W)\***, i.e., which fraction of the are to keep. In our setting, you can obtain the appropriate ***p\*** by picking ***height_min\***, ***height_max\***  ***width_min\***, ***width_max\*** to be all equal to ***sqrt(p)\***
> - *Cropout((height_min,height_max), (width_min,width_max))*, the parameters have the same meaning as in case of *Crop*.
> - *Dropout(keep_min, keep_max)* : where the ratio of the pixels to keep from the watermarked image, ***keep_ratio\***, is drawn uniformly from the range ***(keep_min,keep_max)\***.
> - *Resize(keep_min, keep_max)*, where the resize ratio is drawn uniformly from the range ***(keep_min, keep_max)\***. This ratio applies to both dimensions. For instance, of we have  Resize(0.7, 0.9), and we randomly draw the number 0.8 for a particular  image, then the resulting image will have the dimensions (H * 0.8, W *  0.8).
> - *Jpeg* does not have any parameters.
>
> The Combined-noise
>
> **combined-noise** is the configuration  'crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()'. This is somewhat similar to combined noise configuation in the paper.
- **Crop((0.4,0.55),(0.4,0.55))** means that the height and weight of the cropped image have the expected value of (0.4 + 0.55)/2 = 0.475. Therefore, the ratio of (epected) area of the Cropped image against the original image is 0.475x0.475 ≈ 0.22.
- **Cropout((0.25,0.35),(0.25,0.35))** is similar to Crop(...), this translates to ratio of Cropped vs original image areas with p = 0.09.
- **jpeg()** the same as the Jpeg layer from the paper. It is a differentiable approximation of Jpeg compression with the highest compression coefficient.

I use the images in `datasets/samples` to see the final performance (calculate their PSNR and error ratio).
| EXP_NAME       | avg_err_ratio | avg_psnr |
| -------------- | ------------- | -------- |
| original       | 0             | 22.71    |
| Rel loss       | 0             | 22.79    |
| noise          | 0.41          | 22.65    |
| Rel loss+noise | 0.34          | 22.59    |

`note:` Since the info is random generated, the metrics may be a slight different.

`PSNR:` Peak Signal to Noise Ratio, which can measure the robustness of the model. The higher the PSNR value, the model is more steady to noises. It can be calculated by the following step:
$$MSE=\frac{\sum_{M,N}[I_1(m,n)-I_2(m,n)]^2}{M\times N}(1)$$
where, $M$ and $N$ is the number of rows and columns in the input image respectively. $I$ is the pixel value.
$$PSNR = 10\times\log_{10}{\frac{R^2}{MSE}}(2)$$
where $R$ is the fluctuation in the input image. For example, if the input image has a double-precision floating-point data type, then $R$ is 1. If it has an 8-bit unsigned integer data type, $R$ is 255.

## Notes

`logs/` : This directory contains the log files during the training and testing processes. The logger name is organised in the form "ACTION∈[train/test]+TIME". Each Logger records the detailed loss infomation and error ratio each iter and epoch.

`ckpt/` : This directory contains the model files and test images. I split the *celebaA* dataset into train set(200000 images) and test set(2599 images). For each epoch in training, use two data samples (one fix, one random. Fix choose a fixed info and val set's first image; random choose a random info and a random image in val set) to val the performance. For each epoch in testing, use the whole val set to test the performance. models and test_images are stored each epoch. We can discover that:

- encoded images which can fool the discriminator may can not fool human's eyes.

`datasets/samples` : This directory contains 6 PNG images from Internet and 6 images from *celebaA* which has been converted from JPG to PNG through **opencv-python**. `encode_or_decode.py` is designed for these files to encode and decode a small number of images. The results are stored in `datasets/results`, which sub-directories are named by the model's logger file's name. We can observe that:

- models perform better on train dataset than arbitary images.

You can download these files from 