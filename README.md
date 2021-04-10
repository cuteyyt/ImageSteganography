# Image Steganography

## Introduction

The course project for "Network Security Theory and Practice(网络安全原理与实践)"

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
| noise          | 0.2             | 0.003           |
| Rel loss+noise | 0.162           | 0.003           |

`original:` trains the model using original loss function and without noises.

`Rel loss:` trains the model using RelGAN's loss function and with noises.

 `noise:` trains the model using origial loss function and without noises.

`Rel loss+noise`:

`train_err_ratio:` the error ratio during training, using the train images and training info.

`test_err_ratio:`  the error ratio during testing, using the test images and random generate info.

 `noises:`  the noises combination follows the original repo.

`the noise layer parameters`

>  Noise Layer paremeters
>
> - *Crop((height_min,height_max),(width_min,width_max))*, where ***(height_min,height_max)\*** is a range from which we draw a random number and keep that fraction of the height of the original image. ***(width_min,width_max)\*** controls the same for the width of the image. Put it another way, given an image with dimensions ***H x W,\*** the Crop() will randomly crop this into dimensions ***H' x W'\***, where ***H'/H\*** is in the range ***(height_min,height_max)\***, and ***W'/W\*** is in the range ***(width_min,width_max)\***. In the paper, the authors use a single parameter ***p\*** which shows the ratio ***(H' \* W')/ (H \* W)\***, i.e., which fraction of the are to keep. In our setting, you can obtain the appropriate ***p\*** by picking ***height_min\***, ***height_max\***  ***width_min\***, ***width_max\*** to be all equal to ***sqrt(p)\***
> - *Cropout((height_min,height_max), (width_min,width_max))*, the parameters have the same meaning as in case of *Crop*.
> - *Dropout(keep_min, keep_max)* : where the ratio of the pixels to keep from the watermarked image, ***keep_ratio\***, is drawn uniformly from the range ***(keep_min,keep_max)\***.
> - *Resize(keep_min, keep_max)*, where the resize ratio is drawn uniformly from the range ***(keep_min, keep_max)\***. This ratio applies to both dimensions. For instance, of we have  Resize(0.7, 0.9), and we randomly draw the number 0.8 for a particular  image, then the resulting image will have the dimensions (H * 0.8, W *  0.8).
> - *Jpeg* does not have any parameters.

`the combined-noise`

> **combined-noise** is the configuration  'crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()'. This is somewhat similar to combined noise configuation in the paper.

I use the images in `datasets/samples` to see the final performance (calculate their PSNR and error ratio).

 `PSNR:` Peak Signal to Noise Ratio, which can measure the robustness of the model. The higher the PSNR value, the model is more steady to noises.

