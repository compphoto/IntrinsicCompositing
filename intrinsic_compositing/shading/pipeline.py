import torch
from torch.optim import Adam

import numpy as np

from skimage.transform import resize

from chrislib.general import uninvert, invert, round_32, view

from altered_midas.midas_net import MidasNet

def load_reshading_model(path, device='cuda'):
    
    if path == 'paper_weights':
        state_dict = torch.hub.load_state_dict_from_url('', map_location=device, progress=True)
    else:
        state_dict = torch.load(path) 

    shd_model = MidasNet(input_channels=9)
    shd_model.load_state_dict(state_dict)
    shd_model = shd_model.eval()
    shd_model = shd_model.to(device)

    return shd_model

def spherical2cart(r, theta, phi):
    return [
         r * torch.sin(theta) * torch.cos(phi),
         r * torch.sin(theta) * torch.sin(phi),
         r * torch.cos(theta)
    ]

def run_optimization(params, A, b):
    
    optim = Adam([params], lr=0.01)
    prev_loss = 1000
    
    init_params = params.clone()
    
    for i in range(500):
        optim.zero_grad()

        x, y, z = spherical2cart(params[2], params[0], params[1])

        dir_shd = (A[:, 0] * x) + (A[:, 1] * y) + (A[:, 2] * z)
        pred_shd = dir_shd + params[3]

        loss = torch.nn.functional.mse_loss(pred_shd.reshape(-1), b)

        loss.backward()

        optim.step()

        # theta can range from 0 -> pi/2 (0 to 90 degrees)
        # phi can range from 0 -> 2pi (0 to 360 degrees)
        with torch.no_grad():
            if params[0] < 0:
                params[0] = 0
                
            if params[0] > np.pi / 2:
                params[0] = np.pi / 2
                
            if params[1] < 0:
                params[1] = 0
                
            if params[1] > 2 * np.pi:
                params[1] = 2 * np.pi   
                
            if params[2] < 0:
                params[2] = 0
                
            if params[3] < 0.1:
                params[3] = 0.1
        
        delta = prev_loss - loss
            
        if delta < 0.0001:
            break
            
        prev_loss = loss
        
    return loss, params

def test_init(params, A, b):
    x, y, z = spherical2cart(params[2], params[0], params[1])

    dir_shd = (A[:, 0] * x) + (A[:, 1] * y) + (A[:, 2] * z)
    pred_shd = dir_shd + params[3]

    loss = torch.nn.functional.mse_loss(pred_shd.reshape(-1), b)
    return loss

def get_light_coeffs(shd, nrm, img, mask=None, bias=True):
    img = resize(img, shd.shape)

    reg_shd = uninvert(shd)
    valid = (img.mean(-1) > 0.05) * (img.mean(-1) < 0.95)

    if mask is not None:
        valid *= (mask == 0)
    
    nrm = (nrm * 2.0) - 1.0
    
    A = nrm[valid == 1]
    # A = nrm.reshape(-1, 3)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    
    b = reg_shd[valid == 1]
    # b = reg_shd.reshape(-1)
    
    # parameters are theta, phi, and bias (c)
    A = torch.from_numpy(A)
    b = torch.from_numpy(b)
    
    min_init = 1000
    for t in np.arange(0, np.pi/2, 0.1):
        for p in np.arange(0, 2*np.pi, 0.25):
            params = torch.nn.Parameter(torch.tensor([t, p, 1, 0.5]))
            init_loss = test_init(params, A, b)
    
            if init_loss < min_init:
                best_init = params
                min_init = init_loss
                # print('new min:', min_init)
    
    loss, params = run_optimization(best_init, A, b)
    
    nrm_vis = nrm.copy()
    nrm_vis = draw_normal_circle(nrm_vis, (50, 50), 40)
    
    x, y, z = spherical2cart(params[2], params[0], params[1])

    coeffs = torch.tensor([x, y, z]).reshape(3, 1).detach().numpy()
    out_shd = (nrm_vis.reshape(-1, 3) @ coeffs) + params[3].item()

    coeffs = np.array([x.item(), y.item(), z.item(), params[3].item()])

    return coeffs, out_shd.reshape(shd.shape)

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

def generate_shd(nrm, coeffs, msk, bias=True, viz=False):
    
    # if viz:
        # nrm = draw_normal_circle(nrm.copy(), (50, 50), 40)

    nrm = (nrm * 2.0) - 1.0

    A = nrm.reshape(-1, 3)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    A_fg = nrm[msk == 1]
    A_fg /= np.linalg.norm(A_fg, axis=1, keepdims=True)

    if bias:
        A = np.concatenate((A, np.ones((A.shape[0], 1))), 1)
        A_fg = np.concatenate((A_fg, np.ones((A_fg.shape[0], 1))), 1)
    
    inf_shd = (A_fg @ coeffs)
    inf_shd = inf_shd.clip(0) + 0.2

    if viz:
        shd_viz = (A @ coeffs).reshape(nrm.shape[:2])
        shd_viz = shd_viz.clip(0) + 0.2
        return inf_shd, shd_viz


    return inf_shd

def compute_reshading(orig, msk, inv_shd, depth, normals, alb, coeffs, model):

    # expects no channel dim on msk, shd and depth
    if len(inv_shd.shape) == 3:
        inv_shd = inv_shd[:, :, 0]

    if len(msk.shape) == 3:
        msk = msk[:, :, 0]

    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    h, w, _ = orig.shape

    # max_dim = max(h, w)
    # if max_dim > 1024:
    #     scale = 1024 / max_dim
    # else:
    #     scale = 1.0

    orig = resize(orig, (round_32(h), round_32(w)))
    alb = resize(alb, (round_32(h), round_32(w)))
    msk = resize(msk, (round_32(h), round_32(w)))
    inv_shd = resize(inv_shd, (round_32(h), round_32(w)))
    dpt = resize(depth, (round_32(h), round_32(w)))
    nrm = resize(normals, (round_32(h), round_32(w)))
    msk = msk.astype(np.float32)

    hard_msk = (msk > 0.5)

    reg_shd = uninvert(inv_shd)
    img = (alb * reg_shd[:, :, None]).clip(0, 1)

    orig_alb = orig / reg_shd[:, :, None].clip(1e-4)
    
    bad_shd_np = reg_shd.copy()
    inf_shd = generate_shd(nrm, coeffs, hard_msk)
    bad_shd_np[hard_msk == 1] = inf_shd

    bad_img_np = alb * bad_shd_np[:, :, None]

    sem_msk = torch.from_numpy(msk).unsqueeze(0)
    bad_img = torch.from_numpy(bad_img_np).permute(2, 0, 1)
    bad_shd = torch.from_numpy(invert(bad_shd_np)).unsqueeze(0)
    in_nrm = torch.from_numpy(nrm).permute(2, 0, 1)
    in_dpt = torch.from_numpy(dpt).unsqueeze(0)
    # inp = torch.cat((sem_msk, bad_img, bad_shd), dim=0).unsqueeze(0)
    inp = torch.cat((sem_msk, bad_img, bad_shd, in_nrm, in_dpt), dim=0).unsqueeze(0)
    inp = inp.cuda()
    
    with torch.no_grad():
        out = model(inp).squeeze()

    fin_shd = out.detach().cpu().numpy()
    fin_shd = uninvert(fin_shd)
    fin_img = alb * fin_shd[:, :, None]

    normals = resize(nrm, (h, w))
    fin_shd = resize(fin_shd, (h, w))
    fin_img = resize(fin_img, (h, w))
    bad_shd_np = resize(bad_shd_np, (h, w))

    result = {}
    result['reshading'] = fin_shd
    result['init_shading'] = bad_shd_np
    result['composite'] = (fin_img ** (1/2.2)).clip(0, 1)
    result['normals'] = normals

    return result
