import pathlib
from skimage.transform import resize
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
from argparse import Namespace

from chrislib.general import np_to_pil

from intrinsic_composite.albedo.model.editingnetwork_trainer import EditingNetworkTrainer

PAPER_WEIGHTS_URL = ''
CACHE_PATH = torch.hub.get_dir()


def get_transform(opt, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
        method=Image.BILINEAR
    if 'resize' in opt['preprocess']:
        osize = [opt['load_size'], opt['load_size']]
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in opt['preprocess']:
            transform_list.append(transforms.RandomCrop(opt['crop_size']))

    if not opt['no_flip']:
            transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        transform_list += [transforms.ToTensor()]
        # if grayscale:
        #     transform_list += [transforms.Normalize((0.5,), (0.5,))]
        # else:
        #     transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def match_scalar(source, target, mask=None, min_percentile=0, max_percentile=100):

    if mask is None:
        # mask = np.ones((self.args.crop_size, self.args.crop_size), dtype=bool)
        mask = np.ones((384, 384), dtype=bool)

    target_masked = target[:,mask]


    # consider all values up to a percentile
    p_threshold_min = np.percentile(target_masked.reshape(-1), min_percentile)
    p_mask_min = np.greater_equal(target, p_threshold_min)

    p_threshold_max = np.percentile(target_masked.reshape(-1), max_percentile)
    p_mask_max = np.less_equal(target, p_threshold_max)

    p_mask = np.logical_and(p_mask_max, p_mask_min)
    mask = np.logical_and(p_mask, mask)

    flat_source = source[mask]
    flat_target = target[mask]

    scalar, _, _, _ = np.linalg.lstsq(
        flat_source.reshape(-1, 1), flat_target.reshape(-1, 1), rcond=None)
    source_scaled = source * scalar
    return source_scaled, scalar

def prep_input(rgb_img, mask_img, shading_img):
    # this function takes the srgb image (rgb_img) the mask
    # and the shading as numpy arrays between [0-1]

    opt = {}
    opt['load_size'] = 384
    opt['crop_size'] = 384
    opt['preprocess'] = 'resize'
    opt['no_flip'] = True

    rgb_transform = get_transform(opt, grayscale=False)
    mask_transform = get_transform(opt, grayscale=True)
    shading_transform = get_transform(opt, grayscale=False, method=Image.BILINEAR)

    mask_img_np = mask_img
    if len(mask_img_np.shape) == 3:
       mask_img_np = mask_img_np[:, :, 0] 

    if len(shading_img.shape) == 3:
       shading_img = shading_img[:, :, 0] 

    full_shd = ((1.0 / (shading_img)) - 1.0)
    full_msk = resize(mask_img_np, full_shd.shape)
    # full_msk = resize(np.array(mask_img) / 255., full_shd.shape)
    full_img = resize(rgb_img, full_shd.shape)
    full_alb = (full_img ** 2.2) / full_shd[:, :, None].clip(1e-4)
    full_alb = full_alb.clip(1e-4) ** (1/2.2)

    full_alb = torch.from_numpy(full_alb).permute(2, 0, 1)
    full_shd = torch.from_numpy(full_shd).unsqueeze(0)
    full_msk = torch.from_numpy(full_msk).unsqueeze(0)
    
    srgb = rgb_transform(np_to_pil(rgb_img))
    rgb_mask = np.concatenate([mask_img] * 3, -1)
    mask = mask_transform(np_to_pil(rgb_mask))
    invshading = shading_transform(np_to_pil(shading_img)) # / (2**16-1)

    shading = ((1.0 / invshading) - 1.0)

    ## compute albedo
    rgb = srgb ** 2.2
    albedo = rgb / shading

    ## min max normalize the albedo:
    # albedo = (albedo - albedo.min()) / (albedo.max() - albedo.min())

    ## match the albedo to the rgb
    albedo, scalar = match_scalar(albedo.numpy(),rgb.numpy())
    albedo = torch.from_numpy(albedo)
    albedo = albedo ** (1/2.2) 

    shading = shading / scalar
    ## clip albedo to [0,1]
    albedo = torch.clamp(albedo,0,1)

    
    # all of these need to have a batch dimension as if they are coming from the dataloader
    return {
        'srgb': srgb.unsqueeze(0), 
        'mask': mask.unsqueeze(0), 
        'albedo':albedo.unsqueeze(0), 
        'shading': shading.unsqueeze(0), 
        'albedo_full' : full_alb.unsqueeze(0), 
        'shading_full' : full_shd.unsqueeze(0), 
        'mask_full' : full_msk.unsqueeze(0)
    }


def load_albedo_harmonizer():
    
    # cur_path = pathlib.Path(__file__).parent.resolve()

    args = Namespace()
    args.nops = 4
    args.gpu_ids = [0]
    args.blursharpen = 0
    args.fake_gen_lowdev = 0
    args.bn_momentum = 0.01
    args.edit_loss = ''
    args.loss_relu_bias = 0
    args.crop_size = 384
    args.load_size = 384
    args.lr_d = 0.00001
    args.lr_editnet = 0.00001
    args.batch_size = 1

    args.checkpoint_load_path = f'{CACHE_PATH}/albedo_harmonization/168000_net_Parameters.pth'
    # args.checkpoint_load_path = f'{cur_path}/checkpoints/168000_net_Parameters.pth'
    
    if not os.path.exists(args.checkpoint_load_path):
        os.mkdir(f'{CACHE_PATH}/albedo_harmonization', exists_ok=True)
        os.system(f'wget {PAPER_WEIGHTS_URL} -P {CACHE_PATH}/albedo_harmonization')

    trainer = EditingNetworkTrainer(args)
    return trainer

def harmonize_albedo(img, shd, msk, trainer):
    
    trainer.setEval()

    data = prep_input(img, shd, msk)

    trainer.setinput_HR(data)

    with torch.no_grad():
        trainer.forward()

    albedo_out = trainer.result[0,...].cpu().detach().numpy().squeeze().transpose([1,2,0])
    result = albedo_out.clip(0, 1)
    return result

