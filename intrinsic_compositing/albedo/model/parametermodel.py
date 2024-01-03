import torch
import torch.nn as nn
import torch.nn.functional as F

from  .MiDaS.midas.blocks import _make_pretrained_efficientnet_lite3
from ..utils.networkutils import Conv2dSameExport
from ..utils.utils import normalize

        
class ParametersRegressor(nn.Module):
    def __init__(self, args):
        super(ParametersRegressor,self).__init__()
        
        self.blurandsharpen = args.blursharpen
        self.nf = 384
        self.shared_dec_nfeat = 128
        self.perm_nfeat = 32
        self.nops = args.nops

        self.encoder = _make_pretrained_efficientnet_lite3(use_pretrained=True, exportable=True)
        self.encoder.layer1[0] = Conv2dSameExport(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.encoder.layer4[1][0].bn3 = nn.Identity()

        if args.bn_momentum is not None:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = args.bn_momentum

        self.shared_decoder = nn.Sequential(
            nn.Linear(self.nf  + self.perm_nfeat, self.shared_dec_nfeat),
            nn.LeakyReLU(0.1,False),
            nn.Linear(self.shared_dec_nfeat, self.shared_dec_nfeat),
        )

        self.WB_head = nn.Sequential(
            nn.Linear(self.shared_dec_nfeat, 3),
            nn.Sigmoid(), 
        )
        self.ColorCurve_head = nn.Sequential(
            nn.Linear(self.shared_dec_nfeat, 24),
            nn.Sigmoid(), 
        )
        self.Satur_head = nn.Sequential(
            nn.Linear(self.shared_dec_nfeat, 1),
            nn.Sigmoid(), 
        )
        self.Expos_head = nn.Sequential(
            nn.Linear(self.shared_dec_nfeat, 1),
            nn.Sigmoid(), 
        )

        self.perm_modulation = nn.Sequential(
                nn.Linear(self.nops,self.perm_nfeat)
            )

        if self.blurandsharpen:
            self.Sharp_head = nn.Sequential(
            nn.Linear(self.shared_dec_nfeat, 1),
            nn.Sigmoid(),
            )

            self.Blur_head = nn.Sequential(
                nn.Linear(self.shared_dec_nfeat, 1),
                nn.Sigmoid(),
            )


        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self,x, permutation):
        
        img_feat = self.encoder.layer1(normalize(x))
        img_feat = self.encoder.layer2(img_feat)
        img_feat = self.encoder.layer3(img_feat)
        img_feat = self.encoder.layer4(img_feat)

        img_feat = self.globalavgpool(img_feat).squeeze(-1).squeeze(-1)
        perm_feat = self.perm_modulation(permutation)

        feat = torch.cat((img_feat,perm_feat),dim=1)

        feat = self.shared_decoder(feat)

        wb_param = self.WB_head(feat)*0.9 + 0.1
        colorcurve = self.ColorCurve_head(feat)*1.5 + 0.5
        sat_param = self.Satur_head(feat)*2
        expos_param = self.Expos_head(feat)*1.5 + 0.5

        if self.blurandsharpen:
            sharp_param = self.Sharp_head(feat)*10 + 1 # smaller than one with generate blured image!
            blur_param = self.Blur_head(feat)*5 + 0.001 # cannot be zero -- blur operation fails for zero kernel

            result_dic = {
                'whitebalancing':wb_param,
                'colorcurve':colorcurve,
                'saturation':sat_param, 
                'exposure':expos_param , 
                'blur':blur_param, 
                'sharpness':sharp_param
                }
        else:
            result_dic = {
                'whitebalancing':wb_param,
                'colorcurve':colorcurve,
                'saturation':sat_param, 
                'exposure':expos_param , 
                }
            
        return result_dic
