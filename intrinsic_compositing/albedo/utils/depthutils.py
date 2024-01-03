import sys
import os

curr_path = "/localhome/smh31/Repositories/intrinsic_composite/realismnet/model/"
# OUR
from BoostingMonocularDepth.utils import ImageandPatchs, ImageDataset, generatemask, getGF_fromintegral, calculateprocessingres, rgb2gray,\
    applyGridpatch

# MIDAS
import BoostingMonocularDepth.midas.utils
from BoostingMonocularDepth.midas.models.midas_net import MidasNet
from BoostingMonocularDepth.midas.models.transforms import Resize, NormalizeImage, PrepareForNet

# PIX2PIX : MERGE NET
from BoostingMonocularDepth.pix2pix.options.test_options import TestOptions
from BoostingMonocularDepth.pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

import torch
from torchvision.transforms import Compose
from torchvision.transforms import transforms

import time
import os
import cv2
import numpy as np
import argparse
from argparse import Namespace
import warnings

whole_size_threshold = 3000  # R_max from the paper
GPU_threshold = 1600 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted 

def create_depth_models(device='cuda', midas_path=None, pix2pix_path=None):
    
    # opt = TestOptions().parse()
    opt = Namespace(Final=False, R0=False, R20=False, aspect_ratio=1.0, batch_size=1, checkpoints_dir=f'{curr_path}/BoostingMonocularDepth/pix2pix/checkpoints', colorize_results=False, crop_size=672, data_dir=None, dataroot=None, dataset_mode='depthmerge', depthNet=None, direction='AtoB', display_winsize=256, epoch='latest', eval=False, generatevideo=None, gpu_ids=[0], init_gain=0.02, init_type='normal', input_nc=2, isTrain=False, load_iter=0, load_size=672, max_dataset_size=10000, max_res=float('inf'), model='pix2pix4depth', n_layers_D=3, name='mergemodel', ndf=64, netD='basic', netG='unet_1024', net_receptive_field_size=None, ngf=64, no_dropout=False, no_flip=False, norm='none', num_test=50, num_threads=4, output_dir=None, output_nc=1, output_resolution=None, phase='test', pix2pixsize=None, preprocess='resize_and_crop', savecrops=None, savewholeest=None, serial_batches=False, suffix='', verbose=False)
    # opt = Namespace()
    # opt.gpu_ids = [0]
    # opt.isTrain = False
    # global pix2pixmodel

    pix2pixmodel = Pix2Pix4DepthModel(opt)

    if pix2pix_path == None:
        pix2pixmodel.save_dir = f'{curr_path}/BoostingMonocularDepth/pix2pix/checkpoints/mergemodel'
    else:
        pix2pixmode.save_dir = pix2pix_path

    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    if midas_path == None:
        midas_model_path = f"{curr_path}/BoostingMonocularDepth/midas/model.pt"
    else:
        midas_model_path = midas_path

    # global midasmodel
    midasmodel = MidasNet(midas_model_path, non_negative=True)
    midasmodel.to(device)
    midasmodel.eval()

    return [pix2pixmodel, midasmodel]


def get_depth(img, models, threshold=0.2):

    pix2pixmodel, midasmodel = models

    # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    mask_org = generatemask((3000, 3000))
    mask = mask_org.copy()

    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = threshold

    # print("start processing")

    input_resolution = img.shape

    scale_threshold = 3  # Allows up-scaling with a scale up to 3

    # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
    # supplementary material.
    whole_image_optimal_size, patch_scale = calculateprocessingres(img, 384,
                                                                   r_threshold_value, scale_threshold,
                                                                   whole_size_threshold)

    # print('\t wholeImage being processed in :', whole_image_optimal_size)

    # Generate the base estimate using the double estimation.
    whole_estimate = doubleestimate(img, 384, whole_image_optimal_size, 1024, pix2pixmodel, midasmodel)
    whole_estimate = cv2.resize(whole_estimate, (input_resolution[1], input_resolution[0]), interpolation=cv2.INTER_CUBIC)

    return whole_estimate


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, pix2pixmodel, midasmodel):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, midasmodel)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, midasmodel)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


# Generate a single-input depth estimation
def singleestimate(img, msize, midasmodel):
    if msize > GPU_threshold:
        # print(" \t \t DEBUG| GPU THRESHOLD REACHED", msize, '--->', GPU_threshold)
        msize = GPU_threshold

    return estimatemidas(img, midasmodel, msize)
    # elif net_type == 1:
    #     return estimatesrl(img, msize)
    # elif net_type == 2:
    #     return estimateleres(img, msize)


def estimatemidas(img, midasmodel, msize, device='cuda'):
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction
