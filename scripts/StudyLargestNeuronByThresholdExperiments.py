PROJECT_PATH = '/home/sci/hdai/Projects/SpuriousPruning'
import torch, sys, argparse
sys.path.append(PROJECT_PATH)

from models.resnet_prune import prune_resnet50
from torch.utils.data import DataLoader
from utils.dataset import *
from utils.utils import * 

def main():
    parser = argparse.ArgumentParser(description='A simple program for demonstration purposes.')
    parser.add_argument('--experiment_dataset', type=str, default='waterbirds', help='either celeba or waterbirds')
    parser.add_argument('--celeba_dataset_path', type=str, help='path to the celeba dataset')
    parser.add_argument('--waterbird_dataset_path', type=str, help='path to the waterbirds dataset')
    parser.add_argument('--modification_mode', type=str, default='zero', help='only supports random_init, random_noise, zero, rewind')
    parser.add_argument('--noise_std', type=float, default=0, help='noise standard deviation')
    parser.add_argument('--row_back_n_epochs', type=int, default=0, help='how many epoch we want to roll back in rewind mode')
    parser.add_argument('--modification_percentage', type=float, default=0.05, help='how much percentage of neurons will be modified')
    parser.add_argument('--if_smallest_magnitude', action='store_true')
    parser.add_argument('--dropout_p', type=float, default=0, help='dropout rate')
    args = parser.parse_args()

    experiment_dataset = args.experiment_dataset
    celeba_dataset_path = args.celeba_dataset_path
    waterbird_dataset_path = args.waterbird_dataset_path
    modification_mode = args.modification_mode
    noise_std = args.noise_std
    row_back_n_epochs = args.row_back_n_epochs
    modification_percentage = args.modification_percentage
    if_smallest_magnitude = args.if_smallest_magnitude
    dropout_p = args.dropout_p
    assert experiment_dataset in ['celeba', 'waterbirds'], 'experiment_dataset only supports celeba or waterbirds' 
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
    output_dir = f'experiments/resnet50_{experiment_dataset}_group_unbalanced_erm'

    train_transform = get_transform(target_resolution=target_resolution, train=True, augment_data=True)
    test_transform = get_transform(target_resolution=target_resolution, train=False, augment_data=False)
    training_dataset = ShortcutLearningDataset(basedir=dataset_path, 
                                        split="train",
                                        reweight=False,
                                        transform=test_transform,
                                        device='cuda:0')

    training_dataloader = DataLoader(training_dataset, 
                                    batch_size=batch_size,
                                    shuffle=False, 
                                    num_workers=0)

    testing_dataset = ShortcutLearningDataset(basedir=dataset_path, 
                                        split="test",
                                        reweight=False,
                                        transform=test_transform,
                                        device='cuda:0')
    testing_dataloader = DataLoader(testing_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = prune_resnet50(pretrained=False, num_classes=1000)
    # checkpoint = torch.load(f'resnet50-19c8e357.pth')
    # model.load_state_dict(checkpoint)
    d = model.fc.in_features
    n_classes = training_dataset.n_classes
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=d, out_features=training_dataset.n_classes),
        torch.nn.Softmax(dim=1)  # Add a softmax layer
    )
    checkpoint = torch.load(f'experiments/resnet50_{experiment_dataset}_group_unbalanced_erm/epoch_{n_epochs}_checkpoints.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    original_outcome_string = group_accuracy_evaluation([0,1,2,3], training_dataloader, model)[2:]
    print(original_outcome_string)

    # configuration_list = [
    # ('random_init', 0.02, 0, 1e-5, True, 0),
    # ('random_init', 0.01, 0, 1e-5, True, 0),
    # ('random_init', 0.005, 0, 1e-5, True, 0),
    # ('random_init', 0.02, 0, 2e-5, True, 0),
    # ('random_init', 0.01, 0, 2e-5, True, 0),
    # ('random_init', 0.005, 0, 2e-5, True, 0),
    # ('random_init', 0.02, 0, 3e-5, True, 0),
    # ('random_init', 0.01, 0, 3e-5, True, 0),
    # ('random_init', 0.005, 0, 3e-5, True, 0),
    # ('random_noise', 0.02, 0, 1e-5, True, 0),
    # ('random_noise', 0.01, 0, 1e-5, True, 0),
    # ('random_noise', 0.005, 0, 1e-5, True, 0),
    # ('random_noise', 0.02, 0, 2e-5, True, 0),
    # ('random_noise', 0.01, 0, 2e-5, True, 0),
    # ('random_noise', 0.005, 0, 2e-5, True, 0),
    # ('random_noise', 0.02, 0, 3e-5, True, 0),
    # ('random_noise', 0.01, 0, 3e-5, True, 0),
    # ('random_noise', 0.005, 0, 3e-5, True, 0),
    # ('zero', 0, 0, 0.05, True, 0.8),
    # ('zero', 0, 0, 0.075, True, 0.8),
    # ('zero', 0, 0, 0.1, True, 0.8),
    # ('zero', 0, 0, 0.15, True, 0.8),
    # ('zero', 0, 0, 0.2, True, 0.8),
    # ('zero', 0, 0, 0.4, True, 0.8),
    # ('zero', 0, 0, 0.5, True, 0.8),
    # ('zero', 0, 0, 0.6, True, 0.8),
    # ('rewind', 0, 5, 1e-5, True, 0),
    # ('rewind', 0, 10, 1e-5, True, 0),
    # ('rewind', 0, 5, 2e-5, True, 0),
    # ('rewind', 0, 10, 2e-5, True, 0),
    # ('rewind', 0, 5, 3e-5, True, 0),
    # ('rewind', 0, 10, 3e-5, True, 0)
    # ]

    # for modification_mode, noise_std, row_back_n_epochs, modification_percentage, if_smallest_magnitude, dropout_p in configuration_list:
        
    if if_smallest_magnitude:
        largest_or_smallest = 'smallest'
    else:
        largest_or_smallest = 'largest'
        
    running_times = 1
    if 'random' in modification_mode:
        save_path = f'{output_dir}/{modification_mode}_{modification_percentage}_std{noise_std}_global_{largest_or_smallest}_neuron.txt'
        running_times = 10
    elif 'rewind' in modification_mode:
        save_path = f'{output_dir}/{modification_mode}_{modification_percentage}_{row_back_n_epochs}epochs_back_global_{largest_or_smallest}_neuron.txt'
    else:
        save_path = f'{output_dir}/{modification_mode}_{modification_percentage}_global_{largest_or_smallest}_neuron.txt'
        
    checkpoint = torch.load(f'{output_dir}/epoch_{n_epochs}_checkpoints.pth.tar')
    old_checkpoint = torch.load(f'{output_dir}/epoch_{n_epochs-row_back_n_epochs}_checkpoints.pth.tar')

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    max_neuron_list = []

    with open(save_path, 'w+') as f:
        f.write(f'{original_outcome_string}\n')
        f.write('modified neurons name, index and value\n')

    '''if doing random init or noise, run multiple times for credible results'''
    for _ in range(running_times):
        model.load_state_dict(checkpoint['model_state_dict'])
        model = modify_weights_by_threshold(model, 
                                            modification_percentage=modification_percentage, 
                                            if_smallest_magnitude=if_smallest_magnitude, 
                                            dropout_p=dropout_p,
                                            mode=modification_mode, 
                                            state_dict=old_checkpoint['model_state_dict'], 
                                            noise_std=noise_std)

        outcome_string = group_accuracy_evaluation([0,1,2,3], training_dataloader, model)
        # outcome_string = group_accuracy_evaluation([0,1,2,3], testing_dataloader, model)
        print(outcome_string)

        with open(save_path, 'a') as f:
            f.write(f'{outcome_string}\n')


if __name__ == '__main__':
    main()