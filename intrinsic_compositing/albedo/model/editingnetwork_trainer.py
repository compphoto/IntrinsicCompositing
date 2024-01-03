import numpy as np
import os
import torch
import torch.nn.functional as F

from ..utils.networkutils import init_net, loadmodelweights
from ..utils.edits import apply_colorcurve, apply_exposure, apply_saturation, apply_whitebalancing, get_edits

from .parametermodel import ParametersRegressor
from .discriminator import VOTEGAN


class EditingNetworkTrainer:
    def __init__(self, args):
        
        self.args = args
        self.edits = get_edits(args.nops)
        self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
        self.model_names = ['Parameters']

        self.net_Parameters = init_net(ParametersRegressor(args),args.gpu_ids)
        self.net_Parameters.train()

        # Initial Editing Network weights
        if args.checkpoint_load_path is not None:
            loadmodelweights(self.net_Parameters,args.checkpoint_load_path, self.device) 
        
        if 'realism' in args.edit_loss:
            self.net_D = init_net(VOTEGAN(args), args.gpu_ids)
            self.set_requires_grad(self.net_D, False)
            # Load the realism network weights
            loadmodelweights(self.net_D,args.realism_model_weight_path, self.device) 
            self.net_D.eval()

        # Set the optimizers
        self.optimizer_Parameters = torch.optim.Adam(self.net_Parameters.parameters(), lr=args.lr_editnet)

        # Set the mode for each network

        # Set the loss functions
        self.criterion_L2 = torch.nn.MSELoss()

        # Set the needed constants and parameters
        self.logs = []

        # self.all_permutations = torch.tensor([
        #                         [0,1,2,3],[0,2,1,3],[0,3,1,2],[0,1,3,2],[0,2,3,1],[0,3,2,1],
        #                         [1,0,2,3],[1,2,0,3],[1,3,0,2],[1,0,3,2],[1,2,3,0],[1,3,2,0],
        #                         [2,0,1,3],[2,1,0,3],[2,3,0,1],[2,0,3,1],[2,1,3,0],[2,3,1,0],
        #                         [3,0,1,2],[3,1,0,2],[3,2,0,1],[3,0,2,1],[3,1,2,0],[3,2,1,0]
        #                         ]).float().to(self.device)


    def setEval(self):
        self.net_Parameters.eval()

    def setTrain(self):
        self.net_Parameters.train()

    def setinput(self, input, mergebatch=1):
        self.srgb = input['srgb'].to(self.device)
        self.albedo = input['albedo'].to(self.device)
        self.shading = input['shading'].to(self.device)
        self.mask = input['mask'].to(self.device)

        if mergebatch > 1:
            self.srgb = torch.reshape(self.srgb, (self.args.batch_size, 3, self.args.crop_size, self.args.crop_size))
            self.albedo = torch.reshape(self.albedo, (self.args.batch_size, 3, self.args.crop_size, self.args.crop_size))
            self.shading = torch.reshape(self.shading, (self.args.batch_size, 1, self.args.crop_size, self.args.crop_size))
            self.mask = torch.reshape(self.mask, (self.args.batch_size, 1, self.args.crop_size, self.args.crop_size))

        albedo_edited = self.create_fake_edited(self.albedo)
        self.albedo_fake = (1 - self.mask) * self.albedo + self.mask * albedo_edited
        self.input = torch.cat((self.albedo_fake,self.mask),dim=1).to(self.device)

        # self.numelmask = torch.sum(self.mask,dim=[1,2,3])
    def setinput_HR(self, input):
        self.srgb = input['srgb'].to(self.device)
        self.albedo_fake = input['albedo'].to(self.device)
        self.albedo_full = input['albedo_full'].to(self.device)
        self.shading_full = input['shading_full'].to(self.device)
        self.mask_full = input['mask_full'].to(self.device)
        self.shading = input['shading'].to(self.device)
        self.mask = input['mask'].to(self.device)
        
        self.input = torch.cat((self.albedo_fake, self.mask),dim=1).to(self.device)


    def create_fake_edited(self,rgb):
        # Randomly choose an edit.
        edited = rgb.clone()
        ne = np.random.randint(0, 4)
        perm = torch.randperm(len(self.edits))

        args = self.args
        device = self.device
        
        for i in range(ne):
            edit_id = perm[i]
            if self.args.fake_gen_lowdev == 0:
                wb_param = torch.rand(args.batch_size, 3).to(device)*0.9 + 0.1
                colorcurve = torch.rand(args.batch_size, 24).to(device)*1.5 + 0.5
                
                sat_param = torch.rand(args.batch_size, 1)*2
                sat_param = sat_param.to(device)

                expos_param = torch.rand(args.batch_size, 1)*1.5 + 0.5
                expos_param = expos_param.to(device)

                blur_param = torch.rand(args.batch_size, 1)*5 + 0.0001 # to make sure the blur param is never exactly zero.
                blur_param = blur_param.to(device)

                sharp_param = torch.rand(args.batch_size, 1)*10 + 1
                sharp_param = sharp_param.to(device)

            else:
                wb_param = torch.rand(args.batch_size, 3).to(device)*0.2 + 0.5
                colorcurve = torch.rand(args.batch_size, 24).to(device)*1.5 + 0.5
                
                sat_param = torch.rand(args.batch_size, 1)*1 + 0.5
                sat_param = sat_param.to(device)

                expos_param = torch.rand(args.batch_size, 1)*1 + 0.5
                expos_param = expos_param.to(device)
                
                blur_param = torch.rand(args.batch_size, 1)*2.5 + 0.0001 # to make sure the blur param is never exactly zero.
                blur_param = blur_param.to(device)

                sharp_param = torch.rand(args.batch_size, 1)*5 + 1
                sharp_param = sharp_param.to(device)

            parameters = {
                'whitebalancing':wb_param,
                'colorcurve':colorcurve,
                'saturation':sat_param,
                'exposure':expos_param,
                'blur':blur_param,
                'sharpness':sharp_param
                }


            edited = torch.clamp(self.edits[edit_id.item()](edited,parameters),0,1)

        return edited.detach()

    def forward(self):
        permutation = torch.randperm(len(self.edits)).float().to(self.device)
        params_dic = self.net_Parameters(self.input, permutation.repeat(self.args.batch_size,1))
        # print(params_dic)
        self.logs.append(params_dic)

        # current_rgb = self.albedo_fake
        current_rgb = self.albedo_full

        for ed_in in range(self.args.nops):
            current_edited = torch.clamp(self.edits[permutation[ed_in].item()](current_rgb,params_dic),0,1)
            current_result = (1 - self.mask_full) * current_rgb + self.mask_full * current_edited

            current_rgb = current_result

        self.result = current_result

        self.result_albedo_srgb = self.result ** (2.2)
        self.result_rgb = self.result_albedo_srgb * self.shading_full
        self.result_srgb = torch.clamp(self.result_rgb ** (1/2.2),0,1)

        

    def computeloss_realism(self):
        if self.args.edit_loss == 'realism':
            after = torch.cat((self.result, self.mask), 1)
            after_D_value = self.net_D(after).squeeze(1)
            self.realism_change = 1 - after_D_value
            self.loss_realism = F.relu(self.realism_change - self.args.loss_relu_bias)
            self.loss_g = torch.mean(self.loss_realism) 

        elif self.args.edit_loss == 'realism_relative':
            before = torch.cat((self.albedo, self.mask), 1)
            before_D_value = self.net_D(before).squeeze(1)
            after = torch.cat((self.result, self.mask), 1)
            after_D_value = self.net_D(after).squeeze(1)

            self.realism_change = before_D_value - after_D_value
            self.loss_realism = F.relu(self.realism_change - self.args.loss_relu_bias)
            self.loss_g = torch.mean(self.loss_realism)
            
        elif self.args.edit_loss == 'MSE':
            self.loss_L2 = self.criterion_L2(self.result, self.albedo)
            self.loss_g = torch.mean(self.loss_L2) 




    def optimize_parameters(self):  
        self.optimizer_Parameters.zero_grad()  
        self.computeloss_realism()
        self.loss_g.backward()
        self.optimizer_Parameters.step()   

        for name, p in self.net_Parameters.named_parameters():
            if p.grad is None:
                print(name)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def savemodel(self,iteration, checkpointdir):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (iteration, name)
                save_path = os.path.join(checkpointdir, save_filename)
                net = getattr(self, 'net_' + name)
                if len(self.args.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.args.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
