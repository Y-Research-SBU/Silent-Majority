import torchvision.transforms as transforms
import torch.nn.functional as F
import torch, copy, itertools, random
import numpy as np
import torch.nn as nn
from utils.dataset import *
from torchvision.transforms import InterpolationMode
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, Subset
from collections import OrderedDict


def get_transform(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
#         transform = transforms.Compose([
#             transforms.RandomResizedCrop(
#                 target_resolution,
#                 scale=(0.7, 1.0),
#                 ratio=(0.75, 1.3333333333333333),
#                 interpolation=InterpolationMode.BILINEAR),
#             transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust brightness, contrast, saturation, and hue
#             transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
#             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations
#             transforms.RandomPerspective(distortion_scale=0.2),  # Random perspective transformations
# #             transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
#             transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        # ])
    return transform


def unravel_index(index, shape):
    '''
    unravel an scalar index to a tuple () by give shape list
    '''
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def concat_list(list_of_list):
    return list(itertools.chain.from_iterable(list_of_list))


def l2_noise(inp, norm_ratio):
    '''
    inp: input on which noise is added
    norm_ratio: ratio of norm of noise to that of input
    '''
    noise_2 = torch.normal(0, 1, size=inp.shape).cuda()
    noise_2 *= norm_ratio*inp.norm()/noise_2.norm()
    return noise_2


def dataset_inference(model, dataloader):
    batch_size = dataloader.batch_size
    total_y_true = torch.zeros((len(dataloader.dataset), dataloader.dataset.n_classes))
    total_y_pred = torch.zeros((len(dataloader.dataset), dataloader.dataset.n_classes))
    total_g = torch.zeros((len(dataloader.dataset)))
    total_idx = torch.zeros((len(dataloader.dataset)))

    with torch.no_grad():
        model.eval()
        for j, batch in enumerate(dataloader):
            if 'bert' in model.__class__.__name__.lower():
                idx, x, y_true, g, p = batch
                ids = x[..., 0]
                mask = x[..., 1]
                token_type_ids = x[..., 2]
                y_pred = model(ids, mask, token_type_ids)
            else:
                idx, x, y_true, g, p, _ = batch.values()
                y_pred = model(x)
            total_y_true[j*batch_size:(j+1)*batch_size] = y_true
            total_y_pred[j*batch_size:(j+1)*batch_size] = y_pred
            total_g[j*batch_size:(j+1)*batch_size] = g
            total_idx[j*batch_size:(j+1)*batch_size] = idx
            
    return total_y_true, total_y_pred, total_g, total_idx


def get_poorest_performing_samples_as_dataloader(model, training_dataset, target_groups=None, if_shuffle=True, batch_size=200):
    training_dataloader = DataLoader(training_dataset, 
                                 batch_size=batch_size,
                                 shuffle=False, 
                                 num_workers=0)
    total_y_true, total_y_pred, total_g, total_idx = dataset_inference(model, training_dataloader) 

    if target_groups==None:
        sample_performance = torch.nn.functional.cross_entropy(total_y_pred, total_y_true, reduction='none')
        poorest_performing_idx = torch.argsort(sample_performance, descending=False)[:batch_size]
        poorest_performing_global_idx = total_idx[poorest_performing_idx].long()
        poorest_performing_samples_dataset = Subset(training_dataset, poorest_performing_global_idx)
    else:
        assert isinstance(target_groups, list), "target_groups shall be a list."
        group_target_position = {0:1, 1:1, 2:0, 3:0}
        poorest_performing_global_idx_list = []
        for target_group in target_groups:
            group_global_idx = total_idx[total_g==target_group].long()
            poorest_performing_local_idx = torch.argsort(total_y_pred[group_global_idx][:,group_target_position[target_group]], descending=False)[:batch_size]
            # poorest_performing_global_idx = group_global_idx[poorest_performing_local_idx]
            poorest_performing_global_idx_list.append(group_global_idx[poorest_performing_local_idx])
        poorest_performing_global_idx = torch.cat(poorest_performing_global_idx_list)
        poorest_performing_samples_dataset = Subset(training_dataset, poorest_performing_global_idx)

    poorest_performing_samples_dataloader = DataLoader(poorest_performing_samples_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=if_shuffle, 
                                        num_workers=0)

    return poorest_performing_samples_dataloader


def get_grads_after_loss_change(model, x, y_true, sample_weight, criterion, robustify=False, n_EoT=1):
    '''
    get_grads_after_loss_change
    Args:
        robustify: To get robust estimate of gradients, should we add gaussian noise to input 
        n_EoT: number of steps for Expectation over transformation (gaussian noise)
    Returns:
        grads_dict: dictionary of gradients corresponding to each parameter in the model
    '''
    grads_dict = {}
    n_EoT = 1 if not robustify else n_EoT
    for _ in range (n_EoT):
        if robustify:
            x = x + l2_noise(x, 0.01)

        y_pred = model(x)
        # criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss = criterion(y_pred, y_true).sum()
        # loss = torch.mean(loss*sample_weight)
        loss.backward()

        for name, param in model.named_parameters():
            if name in grads_dict.keys(): 
                grads_dict[name] += param.grad.detach()
            else: 
                grads_dict[name] = copy.deepcopy(param.grad.detach())

        model.zero_grad()

    for name, param in model.named_parameters():
        grads_dict[name] = grads_dict[name]/n_EoT
    
    return grads_dict, y_pred/n_EoT


def get_most_activated_neuron(grads_dict, channel_wise=True, weights_only=True, order=2):
    '''
    channel wise: Remove weights at the channel level versus at the neuron level
    '''
    max_value = 0
#     max_value = torch.zeros((k))
    max_param_name = None
    max_param_index = None
    for param_name in grads_dict.keys():
#         if 'bn' not in name and 'downsample' not in name and 'fc' not in name:
        if True:
            if len(grads_dict[param_name].shape)==4 and channel_wise==True:
                # is this a conv head (channel wise)
    #             signed_grad = signed_grad.sum(dim=(1,2,3)) # [64,1,7,7] param would become [64], which is the input channel dimension
                neuron_grad_norm = torch.linalg.vector_norm(grads_dict[param_name], dim=(1,2,3), ord=order)
            max_norm = neuron_grad_norm.max()  

            if max_norm>max_value:
                if weights_only:
                    if 'weight' in param_name:
                        max_value = max_norm
                        max_param_name = param_name
            #             if channel wise, the max_param_index would be a integer; if neuron wise, the max_param_index would be a tuple
                        max_param_index = unravel_index(neuron_grad_norm.argmax(), neuron_grad_norm.shape)
                else:
                    max_value = max_norm
                    max_param_name = param_name
        #             if channel wise, the max_param_index would be a integer; if neuron wise, the max_param_index would be a tuple
                    max_param_index = unravel_index(neuron_grad_norm.argmax(), neuron_grad_norm.shape)
    
    return max_param_name, max_param_index, max_value


def get_largest_neuron(model, channel_wise=True, weights_only=True, order=2):
    '''
    channel wise: Remove weights at the channel level versus at the neuron level
    '''
    max_value = 0
    max_param_name = None
    max_param_index = None
    for name, param in (model.named_parameters()):
        if 'bn' not in name and 'downsample' not in name and 'fc' not in name :
            if len(param.data.shape)==4 and channel_wise==True:
                neuron_magnitude = torch.linalg.vector_norm(param.data, dim=(1,2,3), ord=order)
            else:
                neuron_magnitude = param.data.abs()
            max_magnitude = neuron_magnitude.max()

            if max_magnitude>max_value:
                if weights_only:
                    if 'weight' in name:
                        max_value = max_magnitude
                        max_param_name = name
            #             if channel wise, the max_param_index would be a integer; if neuron wise, the max_param_index would be a tuple
                        max_param_index = unravel_index(neuron_magnitude.argmax(), neuron_magnitude.shape)
                else:
                    max_value = max_magnitude
                    max_param_name = name
        #             if channel wise, the max_param_index would be a integer; if neuron wise, the max_param_index would be a tuple
                    max_param_index = unravel_index(neuron_magnitude.argmax(), neuron_magnitude.shape)
    
    return max_param_name, max_param_index, max_value


def get_random_neuron(model):
    '''
    channel wise: Remove weights at the channel level versus at the neuron level
    '''
    state_dict = model.state_dict()
    parameter_list = []
    for name, param in (model.named_parameters()):
        if 'bn' not in name and 'downsample' not in name and 'fc' not in name :
            parameter_list.append((name, param.data.shape[0]))

    param_name = random.sample(parameter_list, 1)[0][0]
    param_index = random.randint(0, state_dict[param_name].shape[0])
    param_value = torch.linalg.norm(state_dict[param_name][param_index])
    
    return param_name, param_index, param_value


def modify_weights(model, param_name, param_index=0, mode='zero', state_dict=None, noise_std=1, return_independent_model=True):
    if return_independent_model:
        new_model = copy.deepcopy(model)
    else:
        new_model = model
    state_dict_curr = new_model.state_dict()

    assert mode=='zero' or mode=='rewind' and state_dict!=None or mode=='layer_rewind' and state_dict!=None or mode=='random_init' or mode=='random_noise'
#     new_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in (new_model.named_parameters()):
            if name != param_name: 
                continue
                
            '''
            For conv param, 
            if channel wise, the param_index would be a integer; 
            if node wise, the param_index would be a tuple;
            the shape of param_index would take care of the two mode automatically
            '''
        if mode=='zero': 
            state_dict_curr[param_name][param_index] = 0
        elif mode=='rewind':
                state_dict_curr[param_name][param_index] = state_dict[param_name][param_index].clone()
        elif mode=='layer_rewind':
                state_dict_curr[param_name] = state_dict[param_name].clone()
        elif mode=='random_init':
                state_dict_curr[param_name][param_index] = torch.randn_like(state_dict_curr[param_name][param_index])*noise_std
        elif mode=='random_noise':
                state_dict_curr[param_name][param_index] += torch.randn_like(state_dict_curr[param_name][param_index])*noise_std
    
    new_model.load_state_dict(state_dict_curr)
    return new_model


def modify_weights_by_threshold(model, modification_percentage=0.3, if_smallest_magnitude=True, dropout_p=0., mode='zero', state_dict=None, noise_std=1, return_independent_model=True):
    if return_independent_model:
        new_model = copy.deepcopy(model)
    else:
        new_model = model
    state_dict_curr = new_model.state_dict()
    
    weights = []
    prunable_tensors = []
    for module_name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            weights.append(module.weight.clone().cpu().detach().numpy())
            prunable_tensors.append(module.prune_mask.detach())

    number_of_prunable_weights = torch.sum(torch.tensor([torch.sum(v) for v in prunable_tensors])).cpu().numpy()
    number_of_weights_to_prune_magnitude = np.ceil(modification_percentage * number_of_prunable_weights).astype(int)

    # Create a vector of all the unpruned weights in the model.
    weight_vector = np.concatenate([v.flatten() for v in weights])
    # find the top k smallest neuron to prune
    if if_smallest_magnitude==True:
        threshold = np.sort(np.abs(weight_vector))[min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]
    else:
        threshold = np.sort(np.abs(weight_vector))[-min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

    assert mode=='zero' or mode=='rewind' and state_dict!=None or mode=='random_init' or mode=='random_noise'
#     new_model = copy.deepcopy(model)
    with torch.no_grad():
        for param_name, param in (new_model.named_parameters()):
            '''
            For conv param, 
            if channel wise, the param_index would be a integer; 
            if node wise, the param_index would be a tuple;
            the shape of param_index would take care of the two mode automatically
            '''
            if if_smallest_magnitude:
                mask = torch.abs(state_dict_curr[param_name]) <= threshold
            else:
                mask = torch.abs(state_dict_curr[param_name]) >= threshold
            mask *= torch.bernoulli(torch.ones_like(mask)*(1-dropout_p)).bool()

            if mask.any():
                if mode=='zero': 
                    state_dict_curr[param_name][mask] = 0
                elif mode=='rewind':
                    state_dict_curr[param_name][mask] = state_dict[param_name][mask].clone()
                elif mode=='random_init':
                    random_values = torch.rand(state_dict_curr[param_name].shape)*noise_std
                    state_dict_curr[param_name][mask] = random_values[mask]
                elif mode=='random_noise':
                    random_values = torch.rand(state_dict_curr[param_name].shape)*noise_std
                    state_dict_curr[param_name][mask] += random_values[mask]
    
    new_model.load_state_dict(state_dict_curr)
    return new_model


def get_most_activated_neuron_within_layer(grads_dict, param_name, top_k=1, channel_wise=True):    
    with torch.no_grad():                
        '''
        For conv param, 
        if channel wise, the param_index would be a integer; 
        if neuron wise, the param_index would be a tuple;
        the shape of param_index would take care of the two mode automatically
        '''
        if len(grads_dict[param_name].shape)==4 and channel_wise==True:
            neuron_grad_norm = torch.linalg.vector_norm(grads_dict[param_name], dim=(1,2,3))
        else:
            neuron_grad_norm = grads_dict[param_name].abs().max()
        sorted_values, sorted_indices = torch.sort(neuron_grad_norm)
    
    return sorted_indices[:top_k], sorted_values[:top_k]


def get_largest_neuron_within_layer(model, param_name, top_k=1, channel_wise=True):    
    with torch.no_grad():                
        '''
        For conv param, 
        if channel wise, the param_index would be a integer; 
        if neuron wise, the param_index would be a tuple;
        the shape of param_index would take care of the two mode automatically
        '''
        if len(model.state_dict()[param_name].shape)==4 and channel_wise==True:
            neuron_magnitude = torch.linalg.vector_norm(model.state_dict()[param_name], dim=(1,2,3))
        else:
            neuron_magnitude = model.state_dict()[param_name].abs().max()
        sorted_values, sorted_indices = torch.sort(neuron_magnitude)
        
    return sorted_indices[:top_k], sorted_values[:top_k]


def group_accuracy_dict(y_true, y_pred, groups, group):
    # Filter samples belonging to the specified group
    group_mask = np.where(groups==group)
    label = group//2
    y_true_label_by_group = y_true[group_mask].argmax(axis=1)
    y_pred_label_by_group = y_pred[group_mask].argmax(axis=1)
    y_true_binary_by_group = y_true_label_by_group==label
    y_pred_binary_by_group = y_pred_label_by_group==label

    # Calculate accuracy for the specified group
    precision = precision_score(y_true_binary_by_group, y_pred_binary_by_group)
    recall = recall_score(y_true_binary_by_group, y_pred_binary_by_group, zero_division=0)
    accuracy = accuracy_score(y_true_binary_by_group, y_pred_binary_by_group)
    f1 = f1_score(y_true_binary_by_group, y_pred_binary_by_group)

    return precision, recall, accuracy, f1
    

def group_accuracy_evaluation(target_group_ids, dataloader, model, param_name=''):            
    total_y_true, total_y_pred, total_g, _ = dataset_inference(model, dataloader)

    outcome_string = ''
    if isinstance(target_group_ids, list):
        outcome_string += f'{param_name}'
        for target_group_id in target_group_ids:
            precision, recall, accuracy, f1 = group_accuracy_dict(total_y_true.detach().cpu().numpy(), 
                                                             total_y_pred.detach().cpu().numpy(), 
                                                             total_g.detach().cpu().numpy(), 
                                                             target_group_id)
            loss = torch.nn.functional.cross_entropy(total_y_pred, total_y_true, reduction='mean')
            outcome_string += f', {accuracy:.8f}'
    elif isinstance(target_group_ids, int):
        precision, recall, accuracy, f1 = group_accuracy_dict(total_y_true.detach().cpu().numpy(), 
                                                         total_y_pred.detach().cpu().numpy(), 
                                                         total_g.detach().cpu().numpy(), 
                                                         target_group_ids)
        loss = torch.nn.functional.cross_entropy(total_y_pred, total_y_true, reduction='mean')
        outcome_string = f'Group {int(target_group_ids)} accuracy: {accuracy:.4f}, loss: {loss:.2f}'
    else:
        raise ValueError("target_group_ids should either be int or list.")
        
    return outcome_string


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2*batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def nt_xent(x, features2=None, t=0.5):
#     the normalized temperature-scaled cross entropy loss from https://arxiv.org/abs/2002.05709
    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # negative pair score
    out = torch.cat([out_1, out_2], dim=0)
#     calculate the inner product of every feature pairs, yielding a (2*batch_size)*(2*batch_size) matrix, .t() is transpose function
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

#     mask is also of shape (2*batch_size)*(2*batch_size), with positive pair stands at 0 and negative pair stands at 1
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # positive pair score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss
    loss = -torch.log(pos / (pos + Ng))

    return loss.mean()


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


class Mask(object):
    def __init__(self, model, no_reset=False):
        super(Mask, self).__init__()
        self.model = model
        if not no_reset:
            self.reset()

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""
        prunableTensors = []
        for module_name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in prunableTensors]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

    '''random pruning is element-wise based'''
    def random_pruning(self, random_prune_fraction=0, if_reset=False):
        if if_reset:
            self.reset()
        for module_name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask[torch.rand_like(module.prune_mask) < random_prune_fraction] = 0

    '''magnitude pruning is element-wise based'''
    def magnitude_pruning(self, magnitude_prune_fraction, if_reset=False, if_smallest_magnitude=True):
        # only support one time pruning
        if if_reset:
            self.reset()
        weights = []
        prunable_tensors = [] # for calculating the parameter number
        for module_name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights.append(module.weight.clone().cpu().detach().numpy())
                prunable_tensors.append(module.prune_mask.detach())

        number_of_prunable_weights = torch.sum(torch.tensor([torch.sum(v) for v in prunable_tensors])).cpu().numpy()
        number_of_weights_to_prune_magnitude = np.ceil(magnitude_prune_fraction * number_of_prunable_weights).astype(int)

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v.flatten() for v in weights])
        # find the top k smallest neuron to prune
        if if_smallest_magnitude==True:
            threshold = np.sort(np.abs(weight_vector))[min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]
        else:
            threshold = np.sort(np.abs(weight_vector))[-min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]
            
        pruned_module_dict = {}
        # apply the mask
        for module_name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                # keep the neuron the magnitude of which is larger than certain threshold
                if if_smallest_magnitude==True:
                    module.prune_mask = (torch.abs(module.weight) >= threshold).float()
                else:
                    module.prune_mask = (torch.abs(module.weight) <= threshold).float()
#                 if module.prune_mask.sum()<torch.prod(torch.tensor(module.prune_mask.shape)):
                pruned_module_dict[f'{module_name}.weight'] = round(1-(module.prune_mask.sum()/torch.prod(torch.tensor(module.prune_mask.shape))).item(), 4)
        
        return pruned_module_dict

    '''activation pruning is channel-wise based (if ResNet, else element wise pruning)'''
    def activation_pruning(self, grads_dict, activation_prune_fraction, if_reset=False):
        grads_list = []
        # module_idx_list = []#
        weights_list = []
        # counter = 1
        for module_name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights_list.append(module.weight.clone().cpu().detach().numpy())
                if grads_dict[f'{module_name}.weight'].dim()==4:
                    neuron_grad_norm = torch.linalg.vector_norm(grads_dict[f'{module_name}.weight'], dim=(1,2,3))
                else:
                    neuron_grad_norm = torch.abs(grads_dict[f'{module_name}.weight'])
                grads_list.append(neuron_grad_norm.clone().cpu().detach().numpy())

        # only support one time pruning
        if if_reset:
            self.reset()

        weight_vector = np.concatenate([v.flatten() for v in weights_list])
        sorted_weight_vector = np.sort(np.abs(weight_vector))
        total_weight_number = sorted_weight_vector.shape[0]
        
        number_of_prunable_weights = torch.sum(torch.tensor([np.prod(grad.shape) for grad in grads_list])).cpu().numpy()                    
        number_of_weights_to_prune_activation = np.ceil(activation_prune_fraction*number_of_prunable_weights).astype(int)

        # Create a vector of all the unpruned weights in the model.
        grad_vector = np.concatenate([grad.flatten() for grad in grads_list])
        # np.sort can only sort ascendingly!
        threshold = np.sort(np.abs(grad_vector))[-min(number_of_weights_to_prune_activation, len(grad_vector) - 1)]

        # apply the mask
        pruned_neuron_magnitude_ranking_dict = {}
        for module_name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                if grads_dict[f'{module_name}.weight'].dim()==4:
                    module.prune_mask[torch.linalg.vector_norm(grads_dict[f'{module_name}.weight'], dim=(1,2,3))>=threshold] = 0
                else:
                    module.prune_mask[torch.abs(grads_dict[f'{module_name}.weight'])>=threshold] = 0
#                 record the wiped neuron's index in magnitude ranking
                weight_vector_with_largest_activation = np.abs(module.weight[module.prune_mask==0].flatten().cpu().detach().numpy())
                percentage_index = np.round(np.searchsorted(sorted_weight_vector, weight_vector_with_largest_activation)/total_weight_number, 4)
                pruned_neuron_magnitude_ranking_dict[f'{module_name}.weight'] = percentage_index.tolist()

        return pruned_neuron_magnitude_ranking_dict

    def reset(self):
        for module_name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = torch.ones_like(module.weight)


def save_mask(epoch, model, filename):
    pruneMask = OrderedDict()

    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            pruneMask[name] = module.prune_mask.cpu().type(torch.bool)

    torch.save({"epoch": epoch, "pruneMask": pruneMask}, filename)


def load_mask(model, state_dict, device):
    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            module.prune_mask.data = state_dict[name].to(device).float()

    return model