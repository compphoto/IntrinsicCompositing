import os 
import argparse
from pathlib import Path
import numpy as np

from glob import glob

from skimage.transform import resize

from chrislib.general import (
    invert, 
    uninvert, 
    view, 
    np_to_pil, 
    to2np, 
    add_chan, 
    show, 
    round_32,
    tile_imgs
)
from chrislib.data_util import load_image
from chrislib.normal_util import get_omni_normals

from boosted_depth.depth_util import create_depth_models, get_depth

from intrinsic.model_util import load_models
from intrinsic.pipeline import run_pipeline

from intrinsic_compositing.shading.pipeline import (
    load_reshading_model,
    compute_reshading,
    generate_shd,
    get_light_coeffs
)

from intrinsic_compositing.albedo.pipeline import (
    load_albedo_harmonizer,
    harmonize_albedo
)

from omnidata_tools.model_util import load_omni_model

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rescale(img, scale, r32=False):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]
    
    if r32:
        img = resize(img, (round_32(h * scale), round_32(w * scale)))
    else:
        img = resize(img, (int(h * scale), int(w * scale)))

    return img

def compute_composite_normals(img, msk, model, size):
    
    bin_msk = (msk > 0)

    bb = get_bbox(bin_msk)
    bb_h, bb_w = bb[1] - bb[0], bb[3] - bb[2]

    # create the crop around the object in the image to send through normal net
    img_crop = img[bb[0] : bb[1], bb[2] : bb[3], :]

    crop_scale = 1024 / max(bb_h, bb_w)
    img_crop = rescale(img_crop, crop_scale)
        
    # get normals of cropped and scaled object and resize back to original bbox size
    nrm_crop = get_omni_normals(model, img_crop)
    nrm_crop = resize(nrm_crop, (bb_h, bb_w))

    h, w, c = img.shape
    max_dim = max(h, w)
    if max_dim > size:
        scale = size / max_dim
    else:
        scale = 1.0
    
    # resize to the final output size as specified by input args
    out_img = rescale(img, scale, r32=True)
    out_msk = rescale(msk, scale, r32=True)
    out_bin_msk = (out_msk > 0)
    
    # compute normals for the entire composite image at it's output size
    out_nrm_bg = get_omni_normals(model, out_img)
    
    # now the image is at a new size so the parameters of the object crop change.
    # in order to overlay the normals, we need to resize the crop to this new size
    out_bb = get_bbox(out_bin_msk)
    bb_h, bb_w = out_bb[1] - out_bb[0], out_bb[3] - out_bb[2]
    
    # now resize the normals of the crop to this size, and put them in empty image
    out_nrm_crop = resize(nrm_crop, (bb_h, bb_w))
    out_nrm_fg = np.zeros_like(out_img)
    out_nrm_fg[out_bb[0] : out_bb[1], out_bb[2] : out_bb[3], :] = out_nrm_crop

    # combine bg and fg normals with mask alphas
    out_nrm = (out_nrm_fg * out_msk[:, :, None]) + (out_nrm_bg * (1.0 - out_msk[:, :, None]))
    return out_nrm

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, required=True, help='input directory to read input composites, bgs and masks')
parser.add_argument('--output_dir', type=str, required=True, help='output directory to store harmonized composites')
parser.add_argument('--inference_size', type=int, default=1024, help='size to perform inference (default 1024)')
parser.add_argument('--intermediate', action='store_true', help='whether or not to save visualization of intermediate representations')

args = parser.parse_args()

print('loading depth model')
dpt_model = create_depth_models()

print('loading normals model')
nrm_model = load_omni_model()

print('loading intrinsic decomposition model')
int_model = load_models('paper_weights')

print('loading albedo model')
alb_model = load_albedo_harmonizer()

print('loading reshading model')
shd_model = load_reshading_model('further_trained')


examples = glob(f'{args.input_dir}/*')
print()
print(f'found {len(examples)} scenes')
print()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

for i, example_dir in enumerate(examples):
    
    bg_img = load_image(f'{example_dir}/bg.jpeg')
    comp_img = load_image(f'{example_dir}/composite.png')
    mask_img = load_image(f'{example_dir}/mask.png')

    scene_name = Path(example_dir).stem

    # to ensure that normals are globally accurate we compute them at
    # a resolution of 512 pixels, so resize our shading and image to compute 
    # rescaled normals, then run the lighting model optimization
    bg_h, bg_w = bg_img.shape[:2]
    max_dim = max(bg_h, bg_w)
    scale = 512 / max_dim
    
    small_bg_img = rescale(bg_img, scale)
    small_bg_nrm = get_omni_normals(nrm_model, small_bg_img)
    
    result = run_pipeline(
        int_model,
        small_bg_img ** 2.2,
        resize_conf=0.0,
        maintain_size=True,
        linear=True
    )
    
    small_bg_shd = result['inv_shading'][:, :, None]
    
    
    coeffs, lgt_vis = get_light_coeffs(
        small_bg_shd[:, :, 0], 
        small_bg_nrm, 
        small_bg_img
    )

    # now we compute the normals of the entire composite image, we have some logic
    # to generate a detailed estimation of the foreground object by cropping and 
    # resizing, we then overlay that onto the normals of the whole scene
    comp_nrm = compute_composite_normals(comp_img, mask_img, nrm_model, args.inference_size)

    # now compute depth and intrinsics at a specific resolution for the composite image
    # if the image is already smaller than the specified resolution, leave it
    h, w, c = comp_img.shape
    
    max_dim = max(h, w)
    if max_dim > args.inference_size:
        scale = args.inference_size / max_dim
    else:
        scale = 1.0
    
    # resize to specified size and round to 32 for network inference
    img = rescale(comp_img, scale, r32=True)
    msk = rescale(mask_img, scale, r32=True)
    
    depth = get_depth(img, dpt_model)
    
    result = run_pipeline(
        int_model,
        img ** 2.2,
        resize_conf=0.0,
        maintain_size=True,
        linear=True
    )
    
    inv_shd = result['inv_shading']
    # inv_shd = rescale(inv_shd, scale, r32=True)

    # compute the harmonized albedo, and the subsequent color harmonized image
    alb_harm = harmonize_albedo(img, msk, inv_shd, alb_model) ** 2.2
    harm_img = alb_harm * uninvert(inv_shd)[:, :, None]

    # run the reshading model using the various composited components,
    # and our lighting coefficients computed from the background
    comp_result = compute_reshading(
        harm_img,
        msk,
        inv_shd,
        depth,
        comp_nrm,
        alb_harm,
        coeffs,
        shd_model
    )
    
    if args.intermediate:
        tile_imgs([
            img, 
            msk, 
            1-inv_shd, 
            depth, 
            comp_nrm, 
            view(generate_shd(comp_nrm, coeffs, msk, viz=True)[1]),
            1-invert(comp_result['reshading'])
        ], save=f'{args.output_dir}/{scene_name}_intermediate.jpeg', rescale=0.75)

    np_to_pil(comp_result['composite']).save(f'{args.output_dir}/{scene_name}.png')

    print(f'finished ({i+1}/{len(examples)}) - {scene_name}')
