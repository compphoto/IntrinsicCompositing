import sys
import os
import argparse
import imageio

import torch 
import torch.nn as nn

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from datetime import datetime

from PIL import ImageTk, Image
import numpy as np

from skimage.transform import resize

from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan
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

def viz_coeffs(coeffs, size):
    half_sz = size // 2
    nrm_circ = draw_normal_circle(
        np.zeros((size, size, 3)), 
        (half_sz, half_sz), 
        half_sz
    )
    
    out_shd = (nrm_circ.reshape(-1, 3) @ coeffs[:3]) + coeffs[-1]
    out_shd = out_shd.reshape(size, size)

    lin = np.linspace(-1, 1, num=size)
    ys, xs = np.meshgrid(lin, lin)

    zs = np.sqrt((1.0 - (xs**2 + ys**2)).clip(0))

    out_shd[zs == 0] = 0

    return (out_shd.clip(1e-4) ** (1/2.2)).clip(0, 1)

def draw_normal_circle(nrm, loc, rad):
    size = rad * 2

    lin = np.linspace(-1, 1, num=size)
    ys, xs = np.meshgrid(lin, lin)

    zs = np.sqrt((1.0 - (xs**2 + ys**2)).clip(0))
    valid = (zs != 0)
    normals = np.stack((ys[valid], -xs[valid], zs[valid]), 1)

    valid_mask = np.zeros((size, size))
    valid_mask[valid] = 1

    full_mask = np.zeros((nrm.shape[0], nrm.shape[1]))
    x = loc[0] - rad
    y = loc[1] - rad
    full_mask[y : y + size, x : x + size] = valid_mask
    # nrm[full_mask > 0] = (normals + 1.0) / 2.0
    nrm[full_mask > 0] = normals

    return nrm

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rescale(img, scale):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]

    img = resize(img, (int(h * scale), int(w * scale)))
    return img

def composite_crop(img, loc, fg, mask):
    c_h, c_w, _ = fg.shape 

    img = img.copy()
    
    img_crop = img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w, :]
    comp = (img_crop * (1.0 - mask)) + (fg * mask)
    img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w, :] = comp

    return img

# composite the depth of a fragment but try to match
# the wherever the fragment is placed (fake the depth)
def composite_depth(img, loc, fg, mask):
    c_h, c_w = fg.shape[:2]

    # get the bottom-center depth of the bg
    bg_bc = loc[0] + c_h, loc[1] + (c_w // 2)
    bg_bc_val = img[bg_bc[0], bg_bc[1]].item()

    # get the bottom center depth of the fragment
    fg_bc = c_h - 1, (c_w // 2)
    fg_bc_val = fg[fg_bc[0], fg_bc[1]].item()

    # compute scale to match the fg values to bg
    scale = bg_bc_val / fg_bc_val

    img = img.copy()
    
    img_crop = img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w]
    comp = (img_crop * (1.0 - mask)) + (scale * fg * mask)
    img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w] = comp

    return img


# DISP_SCALE = 0.5
DISP_SCALE = 1.0
MAX_BG_SZ = 960
LGT_VIZ_SZ = 144

# RESHADING_WEIGHTS = '/home/chris/research/intrinsic/data/weights/inpaint_weights/paper_weights.pt'
RESHADING_WEIGHTS = '/home/chris/research/intrinsic/data/weights/inpaint_weights/elegant_cruiser_191_3840.pt'

class App(tk.Tk):
    def __init__(self, args):
        super().__init__()

        self.configure(background='black') 

        self.args = args
        
        loaded_fg = load_image(args.fg)

        self.bg_img = load_image(args.bg)[:, :, :3]
        self.fg_img = loaded_fg[:, :, :3]
        
        if args.mask is not None:
            self.mask_img = load_image(args.mask)
        else:
            if loaded_fg.shape[-1] != 4:
                print("expected foreground image to have an alpha channel since no mask was specified")
                exit()

            self.mask_img = self.fg_img[:, :, -1]

        if len(self.mask_img.shape) == 3:
            self.mask_img = self.mask_img[:, :, :1]
        else:
            self.mask_img = self.mask_img[:, :, np.newaxis]

        
        print('loading depth model')
        self.dpt_model = create_depth_models()

        print('loading normals model')
        self.nrm_model = load_omni_model()

        print('loading intrinsic decomposition model')
        self.int_model = load_models(
            '/home/chris/research/intrinsic/data/weights/final_weights/vivid_bird_318_300.pt',
            '/home/chris/research/intrinsic/data/weights/final_weights/fluent_eon_138_200.pt'
        )

        print('loading albedo model')
        self.alb_model = load_albedo_harmonizer()

        print('loading reshading model')
        self.shd_model = load_reshading_model(
            weights_path=RESHADING_WEIGHTS
        )
        
        self.init_scene()

        self.bg_disp_w = int(self.bg_w * DISP_SCALE)
        self.bg_disp_h = int(self.bg_h * DISP_SCALE)

        win_w = (self.bg_disp_w * 2) + LGT_VIZ_SZ + 40
        win_h = self.bg_disp_h + 20

        # configure the root window
        self.title('compositing demo')
        self.geometry(f"{win_w}x{win_h}")
        # self.geometry(f"")

        self.l_frame = ttk.Frame(self, width=self.bg_disp_w, height=self.bg_disp_h)
        self.l_frame.pack()
        self.l_frame.place(x=10, y=10)

        self.r_frame = ttk.Frame(self, width=self.bg_disp_w, height=self.bg_disp_h)
        self.r_frame.pack()
        self.r_frame.place(x=self.bg_disp_w + 20, y=10)

        style = ttk.Style(self)
        style.configure("TFrame", background="black")

        self.lgt_frame = ttk.Frame(self, width=LGT_VIZ_SZ, height=self.bg_disp_h)
        self.lgt_frame.pack()
        self.lgt_frame.place(x=win_w - LGT_VIZ_SZ - 10, y=10)

        l_disp_img = rescale(self.l_img, DISP_SCALE)
        r_disp_img = rescale(self.r_img, DISP_SCALE)
        lgt_disp_img = viz_coeffs(self.coeffs, LGT_VIZ_SZ)
        
        self.l_photo = ImageTk.PhotoImage(np_to_pil(l_disp_img))
        self.r_photo = ImageTk.PhotoImage(np_to_pil(r_disp_img))
        self.lgt_photo = ImageTk.PhotoImage(np_to_pil(lgt_disp_img))
        
        self.l_label = ttk.Label(self.l_frame, image=self.l_photo)
        self.l_label.pack()

        self.r_label = ttk.Label(self.r_frame, image=self.r_photo)
        self.r_label.pack()

        self.lgt_label = ttk.Label(self.lgt_frame, image=self.lgt_photo)
        self.lgt_label.pack()

        self.bias_scale = ttk.Scale(self.lgt_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, command=self.update_bias)
        self.bias_scale.pack(pady=5)

        style = ttk.Style(self)
        style.configure("White.TLabel", foreground="white", background='black')

        al = ttk.Label(self.lgt_frame, text="Ambient Strength", style="White.TLabel")
        al.pack(pady=5)

        self.dir_scale = ttk.Scale(self.lgt_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, command=self.update_dir)
        self.dir_scale.pack(pady=5)

        dl = ttk.Label(self.lgt_frame, text="Directional Strength", style="White.TLabel")
        dl.pack(pady=5)

        dir_val = np.linalg.norm(self.coeffs[:3])
        bias_val = self.coeffs[-1]

        self.bias_scale.set(bias_val)
        self.dir_scale.set(dir_val)

        self.bind('<Key>', self.key_pressed)
        self.bind('<B1-Motion>', self.click_motion)
        self.bind('<Button-4>', self.scrolled)
        self.bind('<Button-5>', self.scrolled)
        self.bind('<Button-1>', self.clicked)
        
    
    def update_left(self):
        # disp_img = rescale(self.l_img, DISP_SCALE)
        disp_img = self.l_img

        self.l_photo = ImageTk.PhotoImage(np_to_pil(disp_img))
        self.l_label.configure(image=self.l_photo)
        self.l_label.image = self.l_photo

    def update_right(self):
        disp_img = rescale(self.r_img, DISP_SCALE)
        self.r_photo = ImageTk.PhotoImage(np_to_pil(disp_img))
        self.r_label.configure(image=self.r_photo)
        self.r_label.image = self.r_photo

    def update_light(self):

        self.lgt_disp_img = viz_coeffs(self.coeffs, LGT_VIZ_SZ)
        self.lgt_photo = ImageTk.PhotoImage(np_to_pil(self.lgt_disp_img))
        self.lgt_label.configure(image=self.lgt_photo)
        self.lgt_label.image = self.lgt_photo

    
    def update_bias(self, val):
        self.coeffs[-1] = float(val)
        self.update_light()

    def update_dir(self, val):
        vec = self.coeffs[:3]
        vec /= np.linalg.norm(vec).clip(1e-3)
        vec *= float(val)
        self.coeffs[:3] = vec
        self.update_light()

    def init_scene(self):
        bg_h, bg_w, _ = self.bg_img.shape
        
        # resize the background image to be large side < 1024
        max_dim = max(bg_h, bg_w)
        scale = MAX_BG_SZ / max_dim

        self.bg_img = resize(self.bg_img, (int(bg_h * scale), int(bg_w * scale)))
        self.bg_h, self.bg_w, _ = self.bg_img.shape
        
        # compute normals and shading for background, and use them 
        # to optimize for the lighting coefficients
        self.bg_nrm = get_omni_normals(self.nrm_model, self.bg_img)
        result = run_pipeline(
            self.int_model,
            self.bg_img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )

        self.bg_shd = result['inv_shading'][:, :, None]
        self.bg_alb = result['albedo']
        
        max_dim = max(self.bg_h, self.bg_w)
        scale = 512 / max_dim
        small_bg_img = rescale(self.bg_img, scale)
        small_bg_nrm = get_omni_normals(self.nrm_model, small_bg_img)
        small_bg_shd = rescale(self.bg_shd, scale)
        small_bg_img = rescale(self.bg_img, scale)

        self.orig_coeffs, self.lgt_vis = get_light_coeffs(
            small_bg_shd[:, :, 0], 
            small_bg_nrm, 
            small_bg_img
        )

        self.coeffs = self.orig_coeffs

        # first we want to reason about the fg image as an image fragment that we can
        # move, scale and composite, so we only really need to deal with the masked area
        bin_msk = (self.mask_img > 0)

        bb = get_bbox(bin_msk)
        bb_h, bb_w = bb[1] - bb[0], bb[3] - bb[2]

        # create the crop around the object in the image, this can be very large depending
        # on the image that has been chosen by the user, but we want to store it at 1024
        self.orig_fg_crop = self.fg_img[bb[0] : bb[1], bb[2] : bb[3], :].copy()
        self.orig_msk_crop = self.mask_img[bb[0] : bb[1], bb[2] : bb[3], :].copy()
        
        # this is the real_scale, maps the original image crop size to 1024
        max_dim = max(bb_h, bb_w)
        real_scale = MAX_BG_SZ / max_dim
        
        # this is the copy of the crop we keep to compute normals
        self.orig_fg_crop = rescale(self.orig_fg_crop, real_scale)
        self.orig_msk_crop = rescale(self.orig_msk_crop, real_scale)
        
        # now compute the display scale to show the fragment on the ui
        max_dim = max(self.orig_fg_crop.shape)
        disp_scale = (min(self.bg_h, self.bg_w) // 2) / max_dim
        self.frag_scale = disp_scale
        print('init frag_scale:', self.frag_scale)
        
        # these are the versions that the UI shows and what will be 
        # used to create the composite images sent to the networks
        self.fg_crop = rescale(self.orig_fg_crop, self.frag_scale)
        self.msk_crop = rescale(self.orig_msk_crop, self.frag_scale)

        self.bb_h, self.bb_w, _ = self.fg_crop.shape

        self.loc_y = self.bg_h // 2
        self.loc_x = self.bg_w // 2

        top = self.loc_y - (self.bb_h // 2)
        left = self.loc_x - (self.bb_w // 2)

        init_cmp = composite_crop(
            self.bg_img, 
            (top, left),
            self.fg_crop,
            self.msk_crop
        )

        # get normals just for the image fragment, it's best to send it through 
        # cropped and scaled to 1024 in order to get the most details, 
        # then we resize it to match the fragment size
        self.orig_fg_nrm = get_omni_normals(self.nrm_model, self.orig_fg_crop)

        result = run_pipeline(
            self.int_model,
            self.orig_fg_crop ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )

        self.orig_fg_shd = result['inv_shading'][:, :, None]
        self.orig_fg_alb = result['albedo']
        
        del self.nrm_model
        del self.int_model
        torch.cuda.empty_cache()

        self.bg_dpt = get_depth(self.bg_img, self.dpt_model)[:, :, None]
        self.orig_fg_dpt = get_depth(self.orig_fg_crop, self.dpt_model)[:, :, None]
        del self.dpt_model


        self.disp_bg_img = rescale(self.bg_img, DISP_SCALE)
        self.disp_fg_crop = rescale(self.fg_crop, DISP_SCALE)
        self.disp_msk_crop = rescale(self.msk_crop, DISP_SCALE)
        
        self.l_img = init_cmp
        self.r_img = np.zeros_like(init_cmp)

    def scrolled(self, e):

        if e.num == 5 and self.frag_scale > 0.05: # scroll down
            self.frag_scale -= 0.01
        if e.num == 4 and self.frag_scale < 1.0: # scroll up
            self.frag_scale += 0.01
        
        self.fg_crop = rescale(self.orig_fg_crop, self.frag_scale)
        self.msk_crop = rescale(self.orig_msk_crop, self.frag_scale)

        self.disp_fg_crop = rescale(self.fg_crop, DISP_SCALE)
        self.disp_msk_crop = rescale(self.msk_crop, DISP_SCALE)
        
        x = int(self.loc_x * DISP_SCALE)
        y = int(self.loc_y * DISP_SCALE)

        top = y - (self.disp_fg_crop.shape[0] // 2)
        left = x - (self.disp_fg_crop.shape[1] // 2)

        self.l_img = composite_crop(
            self.disp_bg_img, 
            (top, left),
            self.disp_fg_crop,
            self.disp_msk_crop
        )
        self.update_left()
            
    def clicked(self, e):
        x, y = e.x, e.y
        radius = (LGT_VIZ_SZ // 2)

        if e.widget == self.lgt_label:
            rel_x = (x - radius) / radius
            rel_y = (y - radius) / radius
            
            z = np.sqrt(1 - rel_x ** 2 - rel_y ** 2)
            
            print('clicked the lighting viz:', rel_x, rel_y, z)
            
            self.coeffs = np.array([0, 0, 0, float(self.bias_scale.get())])
            dir_vec = np.array([rel_x, -rel_y, z]) * float(self.dir_scale.get())
            self.coeffs[:3] = dir_vec

            self.update_light()

    def click_motion(self, e):
        x, y = e.x, e.y
        
        if e.widget == self.l_label:
            if (x <= self.bg_disp_w) and (y <= self.bg_disp_h):
                
                # we want to show the scaled version of the composite so that the UI
                # can be responsive, but save the coordinates properly so that the 
                # we can send the original size image through the network
                self.loc_y = int(y / DISP_SCALE)
                self.loc_x = int(x / DISP_SCALE)

                top = y - (self.disp_fg_crop.shape[0] // 2)
                left = x - (self.disp_fg_crop.shape[1] // 2)

                self.l_img = composite_crop(
                    self.disp_bg_img, 
                    (top, left),
                    self.disp_fg_crop,
                    self.disp_msk_crop
                )
                
                self.update_left()

    def key_pressed(self, e):
        # run the harmonization
        if e.char == 'r':
            
            # create all the necessary inputs from the state of the interface
            fg_shd_res = rescale(self.orig_fg_shd, self.frag_scale)
            fg_nrm_res = rescale(self.orig_fg_nrm, self.frag_scale)
            fg_dpt_res = rescale(self.orig_fg_dpt, self.frag_scale)

            top = self.loc_y - (self.fg_crop.shape[0] // 2)
            left = self.loc_x - (self.fg_crop.shape[1] // 2)

            self.comp_img = composite_crop(
                self.bg_img, 
                (top, left),
                self.fg_crop,
                self.msk_crop
            )

            self.comp_shd = composite_crop(
                self.bg_shd, 
                (top, left),
                fg_shd_res,
                self.msk_crop
            )

            self.comp_msk = composite_crop(
                np.zeros_like(self.bg_shd), 
                (top, left),
                self.msk_crop,
                self.msk_crop
            )

            comp_nrm = composite_crop(
                self.bg_nrm,
                (top, left),
                fg_nrm_res,
                self.msk_crop
            )

            self.comp_dpt = composite_depth(
                self.bg_dpt,
                (top, left),
                fg_dpt_res,
                self.msk_crop
            )
            
            # the albedo comes out gamma corrected so make it linear
            self.alb_harm = harmonize_albedo(
                self.comp_img, 
                self.comp_msk, 
                self.comp_shd, 
                self.alb_model
            ) ** 2.2
            
            self.orig_alb = (self.comp_img ** 2.2) / uninvert(self.comp_shd)
            harm_img = self.alb_harm * uninvert(self.comp_shd)
            
            self.result = compute_reshading(
                harm_img,
                self.comp_msk,
                self.comp_shd,
                self.comp_dpt,
                comp_nrm,
                self.alb_harm,
                self.coeffs,
                self.shd_model
            )

            self.r_img = self.result['composite']
            self.update_right()

        if e.char == '1':
            self.r_img = self.result['reshading']
            self.update_right()

        if e.char == '2':
            self.r_img = self.result['init_shading']
            self.update_right()

        if e.char == '3':
            self.r_img = self.result['normals']
            self.update_right()

        if e.char == '4':
            self.r_img = self.comp_shd[:, :, 0]
            self.update_right()

        if e.char == '5':
            self.r_img = self.alb_harm
            self.update_right()

        if e.char == '6':
            self.r_img = self.comp_dpt[:, :, 0]
            self.update_right()

        if e.char == 's':
            # save all components 
            
            # orig_shd from intrinsic pipeline is linear and inverse
            orig_shd = add_chan(uninvert(self.comp_shd))

            # reshading coming from compositing pipeline is linear but not inverse
            reshading = add_chan(self.result['reshading'])

            imageio.imwrite('output/orig_shd.exr', orig_shd)
            imageio.imwrite('output/orig_shd.png', orig_shd)
            imageio.imwrite('output/orig_alb.exr', self.orig_alb)
            imageio.imwrite('output/orig_alb.png', self.orig_alb)
            imageio.imwrite('output/orig_img.exr', self.comp_img ** 2.2)
            imageio.imwrite('output/orig_img.png', self.comp_img ** 2.2)

            imageio.imwrite('output/harm_alb.exr', self.alb_harm)
            imageio.imwrite('output/harm_alb.png', self.alb_harm)
            imageio.imwrite('output/reshading.exr', reshading)
            imageio.imwrite('output/reshading.png', reshading)
            imageio.imwrite('output/final.exr', self.result['composite'] ** 2.2)
            imageio.imwrite('output/final.png', self.result['composite'] ** 2.2)

            imageio.imwrite('output/normals.exr', self.result['normals'])
            imageio.imwrite('output/light.exr', self.lgt_disp_img)

        if e.char == 'w':
            # write all the different components as pngs
            
            # orig_shd from intrinsic pipeline is linear and inverse
            orig_shd = add_chan(uninvert(self.comp_shd))

            # reshading coming from compositing pipeline is linear but not inverse
            reshading = add_chan(self.result['reshading'])
            lambertian = add_chan(self.result['init_shading'])
            mask = add_chan(self.comp_msk)
           
            fg_name = Path(self.args.fg).stem
            bg_name = Path(self.args.bg).stem
            ts = int(datetime.utcnow().timestamp())

            save_dir = f'{fg_name}_{bg_name}_{ts}'
            os.makedirs(f'output/{save_dir}')

            np_to_pil(view(orig_shd)).save(f'output/{save_dir}/orig_shd.png')
            np_to_pil(view(lambertian)).save(f'output/{save_dir}/lamb_shd.png')
            np_to_pil(view(self.orig_alb)).save(f'output/{save_dir}/orig_alb.png')
            np_to_pil(self.comp_img).save(f'output/{save_dir}/orig_img.png')

            np_to_pil(view(self.alb_harm)).save(f'output/{save_dir}/harm_alb.png')
            np_to_pil(view(reshading)).save(f'output/{save_dir}/reshading.png')
            np_to_pil(self.result['composite']).save(f'output/{save_dir}/final.png')

            np_to_pil(self.result['normals']).save(f'output/{save_dir}/normals.png')
            np_to_pil(self.lgt_disp_img).save(f'output/{save_dir}/light.png')

            np_to_pil(mask).save(f'output/{save_dir}/mask.png')

            _, bg_lamb_shd = generate_shd(self.bg_nrm, self.coeffs, np.ones(self.bg_nrm.shape[:2]), viz=True)
            np_to_pil(add_chan(view(bg_lamb_shd))).save(f'output/{save_dir}/bg_lamb_shd.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--bg', type=str, required=True)
    parser.add_argument('--fg', type=str, required=True)
    parser.add_argument('--mask', type=str, default=None)

    args = parser.parse_args()
    
    app = App(args)
    app.mainloop()
