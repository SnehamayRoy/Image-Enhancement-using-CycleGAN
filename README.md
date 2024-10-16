# Image-Enhancement-using-CycleGAN

This project uses a CycleGAN (Cycle-Consistent Generative Adversarial Network) to enhance images. CycleGAN is capable of learning image-to-image translation without paired data, which makes it ideal for tasks like image enhancement, style transfer, and domain adaptation.
It is a clean, simple and readable implementation of CycleGAN in PyTorch. I've tried to replicate the original paper as closely as possible.
## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
## Introduction

CycleGAN is a type of generative adversarial network (GAN) that can perform image translation between two domains without the need for paired examples. This project focuses on enhancing images by training a CycleGAN model to translate original into enhanced ones. 

The architecture consists of two GANs:
- One to convert images from domain A (original images) to domain B (enhaced images).
- Another to convert back from domain B to domain A, ensuring cycle consistency.
## Dataset 

The dataset used for this project can consist of original image and enhanced images from various sources such as:

- Custom dataset (if available).
- I have  used the MIT Adobe Fivek dataset [https://data.csail.mit.edu/graphics/fivek/]
- Mainly Expert C and Original images [https://drive.google.com/drive/folders/1x-DcqFVoxprzM4KYGl8SUif8sV-57FP3]
- You can also download it from kaggle-**https://www.kaggle.com/datasets/jesucristo/mit-5k-basic**.
  
You can either prepare your dataset or download one from these source
### Dataset setup(For Training)

- Place your Orginal images to 'data/train/original' directory.
- PLace your Enhaced images to 'data/train/expertc' directory.


# Prerequisites

To get started with this project, you'll need to have the following libraries installed:
- torch
- torchvision
- torchaudio
- albumentations
- tqdm
- numpy
## Installation

Follow these steps to set up the project on your local machine.

### Clone the Repository

First, clone this GitHub repository to your local system:

```bash
git clone https://github.com/SnehamayRoy/Image-Enhancement-using-CycleGAN.git
cd Image-Enhancement-using-CycleGAN
```
Follow these steps to set up the project on colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your_notebook_link.ipynb)
```
!git clone https://github.com/SnehamayRoy/Image-Enhancement-using-CycleGAN.git
!cd Image-Enhancement-using-CycleGAN
```
## Training

- Edit the config.py file to match the setup you want to use. Then run train.py.
  ```bash
  python train.py
  ```
- Make sure 'save.checkpoint=True' in config.py, so that you can use that trained model later.
- You can use the notebook provided here in repository for training 'CycleGAN_for_Training'.
## Testing
- Place your test data respectively 'data\test\expertc' and 'data\test\original' folders .
- Place your trained model into the main folder or you can download my Pretrained model from https://drive.google.com/drive/folders/1z9_PA6VqBZGqMY1cwiS1kBGzdUH8HhxO?usp=sharing (Trained for 120 epochs,Res block=15)
- Then Run test.py
  ```bash
  python test.py
  ```
- You can use the notebook provided here in repository for training 'CycleGAN_for_Testing'.
  
## Results
- Some Generated images will be saved in 'saved_images/'directory during training.
- Test result will be saved in 'test_results'  directory.



