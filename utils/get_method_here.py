import numpy as np
import torch
from PIL import Image
import os


def get_method_here(model_name, weights_path, patch_size=None):
    if False:
        pass
    elif model_name == 'progan':
        model_name = 'progan'
        model_path = os.path.join(weights_path, model_name + '/model_epoch_best.pth')
        arch = 'res50stride1'
        norm_type = 'resnet'
        # patch_size = None
    elif model_name == 'latent':
        model_name = 'latent'
        model_path = os.path.join(weights_path, model_name + '/model_epoch_best.pth')
        arch = 'res50stride1'
        norm_type = 'resnet'
        # patch_size = None
    else:
        print(model_name)
        from get_method import get_method
        model_name, model_path, arch, norm_type, patch_size = get_method(model_name)

    return model_name, model_path, arch, norm_type, patch_size


def rule_minmax(x):
    """
    This function applies a min-max rule to a given input tensor, transforming the tensor based on its mean values.

    Args:
        x (tensor): The input tensor to be transformed.
            * The tensor should have a shape of (batch_size, num_features), where batch_size is the number of samples
            and num_features is the number of features in each sample.
            * Each element of the tensor represents a feature value.

    Returns:
        tensor: A transformed tensor of the same shape as the input.
            * The output tensor is produced by applying the min-max rule based on the mean values of each sample's features.
            * For each sample, if the mean value of its features is less than or equal to 0, the minimum value of the features
            is selected; otherwise, the maximum value of the features is selected.
    """
    x = x.reshape(x.shape[0], 1, -1)
    sm = torch.mean(x, -1)
    su = torch.max(x, -1)[0]
    sd = torch.min(x, -1)[0]
    assert sm.shape == su.shape
    return torch.where(sm <= 0, sd, su)


def rule_trim(x, th=np.log(0.8), tl=np.log(0.2)):
    """
    This function applies a rule-based trimming operation to an input tensor, adjusting values based on given thresholds.

    Args:
        x (tensor): The input tensor to be trimmed.
            * The tensor should have a shape of (batch_size, num_features), where batch_size is the number of samples
            and num_features is the number of features in each sample.
            * Each element of the tensor represents a feature value.
        th (float): The upper threshold for trimming. Values greater than or equal to this threshold are affected.
            * Default value is np.log(0.8).
        tl (float): The lower threshold for trimming. Values less than or equal to this threshold are affected.
            * Default value is np.log(0.2).

    Returns:
        tensor: A transformed tensor of the same shape as the input.
            * The output tensor is produced by applying the trimming rule based on the given thresholds.
            * Values greater than or equal to the upper threshold (th) are trimmed using a ratio of their sum to total count.
            * Values less than or equal to the lower threshold (tl) are trimmed using a similar ratio.
            * Other values are retained.
    """
    x = torch.nn.functional.logsigmoid(x)
    x = x.view(x.shape[0], 1, -1)
    a = torch.median(x, -1)[0]
    su = torch.sum(torch.where(x >= th, x, torch.zeros_like(x)
                               ), -1) / torch.sum(x >= th, -1)
    sd = torch.sum(torch.where(x <= tl, x, torch.zeros_like(x)
                               ), -1) / torch.sum(x <= tl, -1)
    x = torch.mean(x, -1)
    x = torch.where(a >= th, su, x)
    x = torch.where(a <= tl, sd, x)
    return x


dict_rule = {
    'avg': lambda x: torch.mean(x, (-2, -1)),
    'max': lambda x: torch.quantile(x.reshape(x.shape[0], x.shape[1], -1), 1.0, dim=-1),
    'perc97': lambda x: torch.quantile(x.reshape(x.shape[0], x.shape[1], -1), 0.97, dim=-1),
    'minmax': rule_minmax,
    'trim': rule_trim,
    'lss': lambda x: torch.logsumexp(torch.nn.functional.logsigmoid(x), (-2, -1)),
}


def avpool(x, s):
    """
    This function performs average pooling on a 3D input tensor using a specified pooling size.

    Args:
        x (array): The input 3D tensor to be pooled.
            * The tensor should have a shape of (batch_size, channels, height, width), where
            batch_size is the number of samples, channels is the number of input channels,
            height is the input height, and width is the input width.
        s (int): The pooling size, which represents the dimensions of the pooling window.
            * The pooling window is a square of size (s, s).

    Returns:
        array: A pooled 3D tensor of the same number of samples and channels, but with reduced height and width.
            * The output tensor is produced by applying average pooling to non-overlapping windows of size (s, s)
            across the height and width dimensions of the input tensor.
    """
    import scipy.ndimage as ndi
    h = s//2
    return ndi.uniform_filter(x, (1, s, s), mode='constant')[:, h:1-h, h:1-h]


def def_size_avg(arch):
    if arch == 'res50':
        return 8
    elif arch == 'res50stride1':
        return 8
    else:
        assert False


def def_model(arch, model_path, localize=False):
    import torch

    if arch == 'res50':
        # from networks.networks.resnet import resnet50
        from models.resnet import resnet50
        model = resnet50(num_classes=1)
    elif arch == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)
    elif arch == 'res50stride1':
        # import networks.networks.resnet_mod as resnet_mod
        import models.resnet_mod as resnet_mod
        model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
    else:
        print(arch)
        assert False

    assert localize is False

    if model_path == 'None':
        Warning('model_path is None! No weights loading in eval.py')
    else:
        dat = torch.load(model_path, map_location='cpu')
        if 'model' in dat:
            if ('module._conv_stem.weight' in dat['model']) or ('module.fc.fc1.weight' in dat['model']) or ('module.fc.weight' in dat['model']):
                model.load_state_dict(
                    {key[7:]: dat['model'][key] for key in dat['model']})
            else:
                model.load_state_dict(dat['model'])
        elif 'state_dict' in dat:
            model.load_state_dict(dat['state_dict'])
        elif 'net' in dat:
            model.load_state_dict(dat['net'])
        elif 'main.0.weight' in dat:
            model.load_state_dict(dat)
        elif '_fc.weight' in dat:
            model.load_state_dict(dat)
        elif 'conv1.weight' in dat:
            model.load_state_dict(dat)
        else:
            print(list(dat.keys()))
            assert False
    return model
