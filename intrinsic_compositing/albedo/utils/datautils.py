import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data


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