PROJECT_PATH = '/home/sci/hdai/Projects/SpuriousPruning'
import torch, sys, argparse
sys.path.append(PROJECT_PATH)

from torchvision.models import resnet50
from torch.utils.data import DataLoader
from utils.dataset import *
from utils.utils import * 

def main():
    parser = argparse.ArgumentParser(description='A simple program for demonstration purposes.')
    parser.add_argument('--experiment_dataset', type=str, default='waterbirds', help='either celeba or waterbirds')
    parser.add_argument('--experiment_split', type=str, default='train', help='either train or test')
    parser.add_argument('--celeba_dataset_path', type=str, help='path to the celeba dataset')
    parser.add_argument('--waterbird_dataset_path', type=str, help='path to the waterbirds dataset')
    parser.add_argument('--modification_mode', type=str, default='zero', help='only supports random_init, random_noise, zero, rewind')
    parser.add_argument('--noise_std', type=float, default=0, help='noise standard deviation')
    parser.add_argument('--row_back_n_epochs', type=int, default=0, help='how many epoch we want to roll back in rewind mode')
    parser.add_argument('--top_k', type=int, default=3, help='top-k neurons to be pruned')
    args = parser.parse_args()

    experiment_dataset = args.experiment_dataset
    experiment_split = args.experiment_split
    celeba_dataset_path = args.celeba_dataset_path
    waterbird_dataset_path = args.waterbird_dataset_path
    modification_mode = args.modification_mode
    noise_std = args.noise_std
    row_back_n_epochs = args.row_back_n_epochs
    top_k = args.top_k
    assert experiment_dataset in ['celeba', 'waterbirds'], 'experiment_dataset only supports celeba or waterbirds' 
    assert experiment_dataset in ['train', 'test'], 'experiment_split only supports train or test' 
    assert modification_mode in ['random_init', 'random_noise', 'zero', 'rewind'], 'modification_mode only supports random_init, random_noise, zero, rewind' 

    target_resolution = (224, 224)
    batch_size = 200
    n_epochs = 40

    if experiment_dataset=='celeba':
        dataset_path = celeba_dataset_path
        labels = {0:'Non-Blond Female: 71629', 1:'Non-Blond Male: 66874', 2:'Blond Female: 22880', 3:'Blone Male: 1387'}
    elif experiment_dataset=='waterbirds':
        dataset_path = waterbird_dataset_path
        labels = {0:'Land Bird on Land: 3498', 1:'Land Bird on Water: 184', 2:'Water Bird on Land: 56', 3:'Water Bird on Water: 1057'}
    labels = {0:'Land Bird on Land: 3498', 1:'Land Bird on Water: 184', 2:'Water Bird on Land: 56', 3:'Water Bird on Water: 1057'}
    output_dir = f'/experiments/resnet50_{experiment_dataset}_{experiment_split}_group_unbalanced_erm'

    split_transform = get_transform(target_resolution=target_resolution, train=False, augment_data=False)
    target_split_dataset = ShortcutLearningDataset(basedir=dataset_path, 
                                        split=experiment_split,
                                        reweight=False,
                                        transform=split_transform,
                                        device='cuda:1')

    target_split_dataloader = DataLoader(target_split_dataset, 
                                    batch_size=batch_size,
                                    shuffle=False, 
                                    num_workers=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet50()
    d = model.fc.in_features
    n_classes = target_split_dataset.n_classes
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=d, out_features=target_split_dataset.n_classes),
        torch.nn.Softmax(dim=1)  # Add a softmax layer
    )
    checkpoint = torch.load(f'{output_dir}/epoch_{n_epochs}_checkpoints.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    original_outcome_string = group_accuracy_evaluation([0,1,2,3], target_split_dataloader, model)[2:]
    print(original_outcome_string)

    # top_k_list = [2, 3]
    # modification_mode = ['random_init', 'random_noise']
    # noise_std = [0.2, 0.1, 0.05] [0.02,0.01, 0.005]
    # row_back_n_epochs = 5
    # configuration_list = [
    # (1, 'random_init', 0.02, 0),
    # (1, 'random_init', 0.15, 0),
    # (1, 'random_init', 0.1, 0),
    # (2, 'random_init', 0.02, 0),
    # (2, 'random_init', 0.15, 0),
    # (2, 'random_init', 0.1, 0),
    # (3, 'random_init', 0.02, 0),
    # (3, 'random_init', 0.15, 0),
    # (3, 'random_init', 0.1, 0),
    # (1, 'random_noise', 0.02, 0),
    # (1, 'random_noise', 0.01, 0),
    # (1, 'random_noise', 0.005, 0),
    # (2, 'random_noise', 0.02, 0),
    # (2, 'random_noise', 0.01, 0),
    # (2, 'random_noise', 0.005, 0),
    # (3, 'random_noise', 0.02, 0),
    # (3, 'random_noise', 0.01, 0),
    # (3, 'random_noise', 0.005, 0),
    # (1, 'zero', 0, 0),
    # (2, 'zero', 0, 0),
    # (3, 'zero', 0, 0),
    # (4, 'zero', 0, 0),
    # (5, 'zero', 0, 0),
    # (6, 'zero', 0, 0),
    # (7, 'zero', 0, 0),
    # (8, 'zero', 0, 0),
    # (9, 'zero', 0, 0),
    # (10, 'zero', 0, 0),
    # (1, 'rewind', 0, 5),
    # (1, 'rewind', 0, 10),
    # (2, 'rewind', 0, 5),
    # (2, 'rewind', 0, 10),
    # (3, 'rewind', 0, 5),
    # (3, 'rewind', 0, 10)
    # ]

    # for top_k, modification_mode, noise_std, row_back_n_epochs in configuration_list:
    running_times = 1
    if 'random' in modification_mode:
        save_path = f'{output_dir}/{modification_mode}_std{noise_std}_global_top{top_k}_largest_neuron.txt'
        running_times = 10
    elif 'rewind' in modification_mode:
        save_path = f'{output_dir}/{modification_mode}_{row_back_n_epochs}epochs_back_global_top{top_k}_largest_neuron.txt'
    else:
        save_path = f'{output_dir}/{modification_mode}_global_top{top_k}_largest_neuron.txt'
        
    checkpoint = torch.load(f'{output_dir}/epoch_{n_epochs}_checkpoints.pth.tar', map_location=device)
    old_checkpoint = torch.load(f'{output_dir}/epoch_{n_epochs-row_back_n_epochs}_checkpoints.pth.tar', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    max_neuron_list = []

    with open(save_path, 'w+') as f:
        f.write(f'{original_outcome_string}\n')
        f.write('modified neurons name, index and value\n')
    
    '''make sure the second modification won't happen on the same neuron again'''
    for _ in range(top_k):
        model.zero_grad()
        max_param_name, max_param_index, max_param_value = get_largest_neuron(model, channel_wise=True, weights_only=True)
        max_neuron_list.append((max_param_name, max_param_index, max_param_value))
        
        print(max_param_name, max_param_index[0].item(), max_param_value.item())
        with open(save_path, 'a') as f:
            f.write(f'{max_param_name}, {max_param_index[0].item()}, {max_param_value.item()}\n')
            
        '''below will do zero out'''
        model = modify_weights(model=model, 
                            param_name=max_param_name, 
                            param_index=max_param_index)

    '''if doing random init or noise, run multiple times for credible results'''
    for _ in range(running_times):
        model.load_state_dict(checkpoint['model_state_dict'])
        for (max_param_name, max_param_index, max_param_value) in max_neuron_list:
            model = modify_weights(model=model, 
                                param_name=max_param_name, 
                                param_index=max_param_index, 
                                mode=modification_mode, 
                                state_dict=old_checkpoint['model_state_dict'], 
                                noise_std=noise_std)

        outcome_string = group_accuracy_evaluation([0,1,2,3], target_split_dataloader, model)
        print(outcome_string)

        with open(save_path, 'a') as f:
            f.write(f'{outcome_string}\n')


if __name__ == '__main__':
    main()