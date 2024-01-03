import torch.nn as nn


from .pix2pix.models.networks import NLayerDiscriminator,get_norm_layer        
from ..utils.utils import normalize

class VOTEGAN(nn.Module):
    def __init__(self, args):
        super(VOTEGAN, self).__init__()
        n_ic = 4 
        if args.crop_size == 384:
            n_f = 22*22
        else:
            raise NotImplementedError

        ndf = 64 # of discrim filters in the first conv layer
        norm_layer = get_norm_layer(norm_type='instance')
        self.model = nn.Sequential(NLayerDiscriminator(n_ic, ndf, n_layers=4, norm_layer=norm_layer), # 70*70 patchgan for 256*256
        nn.Flatten(start_dim=1),
        nn.Linear(n_f,32),
        nn.Linear(32,1),
        nn.Sigmoid()
        )


    def forward(self,x):
        feat = self.model(normalize(x))
        return feat
