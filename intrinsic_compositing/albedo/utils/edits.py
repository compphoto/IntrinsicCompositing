import torch
import torch.nn as nn
from kornia.color import rgb_to_hsv, hsv_to_rgb

import torchvision.transforms.functional as F
# from ..argumentsparser import args 

COLORCURVE_L = 8

def apply_whitebalancing(input, parameters):
    param = parameters['whitebalancing']
    param = param / (param[:,1:2] + 1e-9)
    result = input / (param[:,:,None,None] + 1e-9)
    return result

def apply_colorcurve(input, parameters):
    color_curve_param = torch.reshape(parameters['colorcurve'],(-1,3,COLORCURVE_L))
    color_curve_sum = torch.sum(color_curve_param,dim=[2])
    total_image = torch.zeros_like(input)
    for i in range(COLORCURVE_L):
        total_image += torch.clip(input * COLORCURVE_L - i, 0, 1) * color_curve_param[:,:,i][:,:,None,None]
    result =  total_image / (color_curve_sum[:,:,None,None] + 1e-9)
    return result

def apply_saturation(input, parameters):
    hsv = rgb_to_hsv(input)
    param = parameters['saturation'][:,:,None,None]
    s_new = hsv[:,1:2,:,:] * param
    hsv_new = hsv.clone()
    hsv_new[:,1:2,:,:] = s_new
    result = hsv_to_rgb(hsv_new)
    return result

def apply_exposure(input, parameters):
    result = input * parameters['exposure'][:,:,None,None]
    return result


def apply_blur(input, parameters):
    sigma = parameters['blur'][:,:,None,None]
    kernelsize = 2*torch.ceil(2*sigma)+1.

    result = torch.zeros_like(input)
    for B in range(input.shape[0]):
        kernelsize_ = (int(kernelsize[B].item()), int(kernelsize[B].item()))
        sigma_ = (sigma[B].item(), sigma[B].item())
        result[B,:,:,:] = F.gaussian_blur(input[B:B+1,:,:,:], kernelsize_, sigma_)
    return result

def apply_sharpness(input, parameters):
    param = parameters['sharpness'][:,0]
    result = torch.zeros_like(input)

    for B in range(input.shape[0]):
        result[B,:,:,:] = F.adjust_sharpness(input[B:B+1,:,:,:], param[B])
    return result

def get_edits(nops):
    if nops == 4:  
        return {
            0: apply_whitebalancing,
            1: apply_colorcurve,
            2: apply_saturation,
            3: apply_exposure,
        }
    else:
        return {
            0: apply_whitebalancing,
            1: apply_colorcurve,
            2: apply_saturation,
            3: apply_exposure,
            4: apply_blur,
            5: apply_sharpness,
        }
