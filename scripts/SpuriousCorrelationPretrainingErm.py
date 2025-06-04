PROJECT_PATH = '/home/sci/hdai/Projects/SpuriousPruning'
import torch, os, json, sys, argparse
sys.path.append(PROJECT_PATH)
import matplotlib.pyplot as plt

from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from models.deit_prune import VisionTransformer
from utils.dataset import *
from utils.utils import *

def main():
    parser = argparse.ArgumentParser(description='A simple program for demonstration purposes.')
    parser.add_argument('--experiment_dataset', type=str, default='waterbirds', help='either celeba or waterbirds')
    parser.add_argument('--experiment_arch', type=str, default='resnet50', help='either resnet50 or deit_tiny, deit_small, deit_base, or deit_large')
    parser.add_argument('--celeba_dataset_path', type=str, help='path to the celeba dataset')
    parser.add_argument('--waterbird_dataset_path', type=str, help='path to the waterbirds dataset')
    args = parser.parse_args()

    experiment_dataset = args.experiment_dataset
    experiment_arch = args.experiment_arch
    celeba_dataset_path = args.celeba_dataset_path
    waterbird_dataset_path = args.waterbird_dataset_path
    assert experiment_dataset in ['celeba', 'waterbirds'], 'experiment_dataset only supports celeba or waterbirds' 
    assert experiment_arch in ['resnet50', 'deit_tiny', 'deit_small', 'deit_base', 'deit_large'], 'experiment_arch only support resnet50, deit_tiny, deit_small, deit_base, or deit_large.' 

    target_resolution = (224, 224)
    batch_size = 256
    mode = 'gpu'

    if mode=='gpu':
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(1)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_dtype(torch.float32)

    augmented_transform = get_transform(target_resolution=target_resolution, train=True, augment_data=True)
    standard_transform = get_transform(target_resolution=target_resolution, train=False, augment_data=False)

    ratio = {}
    if experiment_dataset=='celeba':
        dataset_path = celeba_dataset_path
        training_set_labels = {0:'Non-Blond Female: 71629', 1:'Non-Blond Male: 66874', 2:'Blond Female: 22880', 3:'Blone Male: 1387'}
        testing_set_labels = {0:'Non-Blond Female: 9767', 1:'Non-Blond Male: 7535', 2:'Blond Female: 2480', 3:'Blone Male: 180'}
        validating_set_labels = {0:'Non-Blond Female: 8535', 1:'Non-Blond Male: 8276', 2:'Blond Female: 2874', 3:'Blone Male: 182'}
        ratio['training'] = np.asarray([71629, 66874, 22880, 1387]) # [0.44, 0.41, 0.14, 0.01]
        ratio['testing'] = np.asarray([9767, 7535, 2480, 180]) # [0.49, 0.38, 0.12, 0.01]
        ratio['validating'] = np.asarray([8535, 8276, 2874, 182]) # [0.43, 0.42, 0.14, 0.01]
    elif experiment_dataset=='waterbirds':
        dataset_path = waterbird_dataset_path
        training_set_labels = {0:'Land Bird on Land: 3498', 1:'Land Bird on Water: 184', 2:'Water Bird on Land: 56', 3:'Water Bird on Water: 1057'}
        testing_set_labels = {0:'Land Bird on Land: 2255', 1:'Land Bird on Water: 2255', 2:'Water Bird on Land: 642', 3:'Water Bird on Water: 642'}
        validating_set_labels = {0:'Land Bird on Land: 467', 1:'Land Bird on Water: 466', 2:'Water Bird on Land: 133', 3:'Water Bird on Water: 133'}
        ratio['training'] = np.asarray([3498, 184, 56, 1057]) # [0.73, 0.04, 0.01, 0.22]
        ratio['testing'] = np.asarray([2255, 2255, 642, 642]) # [0.39, 0.39, 0.11, 0.11]
        ratio['validating'] = np.asarray([467, 466, 133, 133]) # [0.39, 0.39, 0.11, 0.11]

    training_dataset = ShortcutLearningDataset(basedir=dataset_path, 
                                        split="train",
                                        reweight=False,
                                        transform=augmented_transform,
                                        device=device)
    training_dataloader = DataLoader(training_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True, 
                                    num_workers=0)

    testing_dataset = ShortcutLearningDataset(basedir=dataset_path, 
                                        split="test",
                                        reweight=False,
                                        transform=standard_transform,
                                        device=device)
    testing_dataloader = DataLoader(testing_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=0)

    validating_dataset = ShortcutLearningDataset(basedir=dataset_path, 
                                        split="val",
                                        reweight=False,
                                        transform=standard_transform,
                                        device=device)
    validating_dataloader = DataLoader(validating_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=0)

    n_epochs = 101
    start_epoch_num = 0
    save_model = True
    resume = False
    log_test_interval = 1
    learning_rate = 5e-6#1e-6
    training_loss_list, testing_loss_list = [], []
    output_dir = f'experiments/{experiment_arch}_from_scratch_{experiment_dataset}_group_unbalanced_erm_nocolorjit_{learning_rate}'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if experiment_arch=='resnet50':
        # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = resnet50()
        d = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=d, out_features=training_dataset.n_classes),
            torch.nn.Softmax(dim=1)  # Add a softmax layer
        )
    else:
        if experiment_arch=='deit_tiny':
            model = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=1000) # deit_tiny_patch16_224
            pretrained_checkpoint_path = 'experiments/deit_tiny_patch16_224-a1311bcf.pth'
        elif experiment_arch=='deit_small':
            model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=1000) # deit_small_patch16_224
            pretrained_checkpoint_path = 'experiments/deit_small_patch16_224-cd65a155.pth'
        elif experiment_arch=='deit_base':
            model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=1000) # deit_base_patch16_224
            pretrained_checkpoint_path = 'experiments/deit_base_patch16_224-b5f2ef4d.pth'
        elif experiment_arch=='deit_large':
            model = VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_classes=21843) # deit_large_patch16_224
            pretrained_checkpoint_path = 'experiments/jx_vit_large_patch16_224_in21k-606da67d.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)
        d = model.head.in_features
        model.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=d, out_features=training_dataset.n_classes),
            torch.nn.Softmax(dim=1)  # Add a softmax layer
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        checkpoint = torch.load(f'{output_dir}/epoch_{start_epoch_num-1}_checkpoints.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_epoch_num = 0
        
    training_y_pred_evolution = torch.zeros((n_epochs, len(training_dataset.metadata_df), 2))
    training_y_true_evolution = torch.zeros((n_epochs, len(training_dataset.metadata_df), 2))
    training_g_evolution = torch.zeros((n_epochs, len(training_dataset.metadata_df)))
    training_idx_evolution = torch.zeros((n_epochs, len(training_dataset.metadata_df)))
    testing_y_pred_evolution = torch.zeros((n_epochs, len(testing_dataset.metadata_df), 2))
    testing_y_true_evolution = torch.zeros((n_epochs, len(testing_dataset.metadata_df), 2))
    testing_g_evolution = torch.zeros((n_epochs, len(testing_dataset.metadata_df)))
    testing_idx_evolution = torch.zeros((n_epochs, len(testing_dataset.metadata_df)))

    for epoch in tqdm(range(start_epoch_num, start_epoch_num+n_epochs)):    
        '''evaluation before training'''
        (training_y_true_evolution[epoch-start_epoch_num], 
        training_y_pred_evolution[epoch-start_epoch_num], 
        training_g_evolution[epoch-start_epoch_num], 
        training_idx_evolution[epoch-start_epoch_num]) = dataset_inference(model, training_dataloader)
        (testing_y_true_evolution[epoch-start_epoch_num], 
        testing_y_pred_evolution[epoch-start_epoch_num], 
        testing_g_evolution[epoch-start_epoch_num], 
        testing_idx_evolution[epoch-start_epoch_num]) = dataset_inference(model, testing_dataloader)
                
        torch.save(training_y_pred_evolution, f'{output_dir}/training_y_pred_evolution.pt')
        torch.save(training_y_true_evolution, f'{output_dir}/training_y_true_evolution.pt')
        torch.save(training_g_evolution, f'{output_dir}/training_g_evolution.pt')
        torch.save(training_idx_evolution, f'{output_dir}/training_idx_evolution.pt')
        torch.save(testing_y_pred_evolution, f'{output_dir}/testing_y_pred_evolution.pt')
        torch.save(testing_y_true_evolution, f'{output_dir}/testing_y_true_evolution.pt')
        torch.save(testing_g_evolution, f'{output_dir}/testing_g_evolution.pt')
        torch.save(testing_idx_evolution, f'{output_dir}/testing_idx_evolution.pt')
                
        epoch_loss = 0
        model.train()
        for i, batch in tqdm(enumerate(training_dataloader)):
            global_id, x, y_true, g, p, weights = batch.values()
            y_pred = model(x)
            
            optimizer.zero_grad()
            training_loss = criterion(y_pred, y_true)
            training_loss = torch.mean(training_loss*weights)
            training_loss.backward()
            optimizer.step()
            epoch_loss += training_loss.item()
        
        training_loss_list.append(epoch_loss)
        scheduler.step(epoch_loss)

        print(f'epoch {epoch} {criterion.__class__.__name__}: {epoch_loss}')
        if epoch%log_test_interval==0: 
            if save_model:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, f'{output_dir}/epoch_{epoch}_checkpoints.pth.tar')

        with open(f'{output_dir}/training_loss_list.json', 'w') as json_file:
            json.dump(training_loss_list, json_file)

    '''B. Group-wise trend in training set'''

    training_y_pred_evolution = torch.load(f'{output_dir}/training_y_pred_evolution.pt', map_location=device)
    training_y_true_evolution = torch.load(f'{output_dir}/training_y_true_evolution.pt', map_location=device)
    training_g_evolution = torch.load(f'{output_dir}/training_g_evolution.pt', map_location=device)

    group_accuracy_dict = {0:[], 1:[], 2:[], 3:[]}

    for i in range(n_epochs):
        for j in range(len(training_set_labels)):
            group_mask = training_g_evolution[i]==j
            y_true_by_instance_group = training_y_true_evolution[i, group_mask].argmax(axis=-1)
            y_pred_by_instance_group = training_y_pred_evolution[i, group_mask].argmax(axis=-1)

            accuracy = accuracy_score(y_true_by_instance_group.cpu(), y_pred_by_instance_group.cpu())
            group_accuracy_dict[j].append(accuracy)
            
    plt.figure(figsize=(15,5))
    full_group_accuracy_string = ''
    for i in range(len(training_set_labels)):
        plt.plot(group_accuracy_dict[i], label=training_set_labels[i])
        group_accuracy_string = f'group{i}'
        for j in range(len(group_accuracy_dict[i])):
            group_accuracy_string += f', {group_accuracy_dict[i][j]:.4f}'
        full_group_accuracy_string += group_accuracy_string+'\n'

    group_accuracy_array = np.array([group_accuracy_dict[i] for i in range(4)])
    average_group_accuracy = (ratio['training'][0]*group_accuracy_array[0]+ratio['training'][1]*group_accuracy_array[1]+ratio['training'][2]*group_accuracy_array[2]+ratio['training'][3]*group_accuracy_array[3])/ratio['training'].sum()
    full_group_accuracy_string += 'averag, ' + ', '.join([f'{x:.4f}' for x in average_group_accuracy]) + '\n'
        
    print(full_group_accuracy_string)
    with open(f'{output_dir}/group_accuracy_evolution_in_training_set.txt', 'w') as file:
        file.write(full_group_accuracy_string)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_accuracy_evolution_in_training_set.png', dpi=300)
    plt.savefig(f'{output_dir}/group_accuracy_evolution_in_training_set.pdf', dpi=300)

    group_loss = {0:[], 1:[], 2:[], 3:[]}
    for i in range(n_epochs):
        for j in range(len(training_set_labels)):
            group_mask = training_g_evolution[i]==j
            y_true_by_instance_group = training_y_true_evolution[i, group_mask]
            y_pred_by_instance_group = training_y_pred_evolution[i, group_mask]

            loss = torch.nn.functional.cross_entropy(y_pred_by_instance_group, y_true_by_instance_group).item()
            group_loss[j].append(loss)
            
    plt.figure(figsize=(15,5))
    for i in range(len(training_set_labels)):
        plt.plot(group_loss[i], label=training_set_labels[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_loss_evolution_in_training_set.png', dpi=300)
    plt.savefig(f'{output_dir}/group_loss_evolution_in_training_set.pdf', dpi=300)

    '''Group-wise trend in testing set'''

    testing_y_pred_evolution = torch.load(f'{output_dir}/testing_y_pred_evolution.pt', map_location=device)
    testing_y_true_evolution = torch.load(f'{output_dir}/testing_y_true_evolution.pt', map_location=device)
    testing_g_evolution = torch.load(f'{output_dir}/testing_g_evolution.pt', map_location=device)

    group_accuracy_dict = {0:[], 1:[], 2:[], 3:[]}

    for i in range(n_epochs):
        for j in range(len(testing_set_labels)):
            group_mask = testing_g_evolution[i]==j
            y_true_by_instance_group = testing_y_true_evolution[i, group_mask].argmax(axis=-1)
            y_pred_by_instance_group = testing_y_pred_evolution[i, group_mask].argmax(axis=-1)

            accuracy = accuracy_score(y_true_by_instance_group.cpu(), y_pred_by_instance_group.cpu())
            group_accuracy_dict[j].append(accuracy)
            
    plt.figure(figsize=(15,5))
    full_group_accuracy_string = ''
    for i in range(len(testing_set_labels)):
        plt.plot(group_accuracy_dict[i], label=testing_set_labels[i])
        group_accuracy_string = f'group{i}'
        for j in range(len(group_accuracy_dict[i])):
            group_accuracy_string += f', {group_accuracy_dict[i][j]:.4f}'
        full_group_accuracy_string += group_accuracy_string+'\n'

    group_accuracy_array = np.array([group_accuracy_dict[i] for i in range(4)])
    average_group_accuracy = (ratio['training'][0]*group_accuracy_array[0]+ratio['training'][1]*group_accuracy_array[1]+ratio['training'][2]*group_accuracy_array[2]+ratio['training'][3]*group_accuracy_array[3])/ratio['training'].sum()
    full_group_accuracy_string += 'averag, ' + ', '.join([f'{x:.4f}' for x in average_group_accuracy]) + '\n'
        
    print(full_group_accuracy_string)
    with open(f'{output_dir}/group_accuracy_evolution_in_testing_set.txt', 'w') as file:
        file.write(full_group_accuracy_string)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_accuracy_evolution_in_testing_set.png', dpi=300)
    plt.savefig(f'{output_dir}/group_accuracy_evolution_in_testing_set.pdf', dpi=300)

    group_loss = {0:[], 1:[], 2:[], 3:[]}

    for i in range(n_epochs):
        for j in range(len(testing_set_labels)):
            group_mask = testing_g_evolution[i]==j
            y_true_by_instance_group = testing_y_true_evolution[i, group_mask]
            y_pred_by_instance_group = testing_y_pred_evolution[i, group_mask]

            loss = torch.nn.functional.cross_entropy(y_pred_by_instance_group, y_true_by_instance_group).item()
            group_loss[j].append(loss)
            
    plt.figure(figsize=(15,5))
    for i in range(len(testing_set_labels)):
        plt.plot(group_loss[i], label=testing_set_labels[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_loss_evolution_in_testing_set.png', dpi=300)
    plt.savefig(f'{output_dir}/group_loss_evolution_in_testing_set.pdf', dpi=300)

    '''Group-wise trend in validation set'''

    validating_y_pred_evolution = torch.zeros((n_epochs, len(validating_dataset.metadata_df), 2))
    validating_y_true_evolution = torch.zeros((n_epochs, len(validating_dataset.metadata_df), 2))
    validating_g_evolution = torch.zeros((n_epochs, len(validating_dataset.metadata_df)))
    validating_idx_evolution = torch.zeros((n_epochs, len(validating_dataset.metadata_df)))

    for epoch in tqdm(range(n_epochs)):
        checkpoint = torch.load(f'{output_dir}/epoch_{epoch}_checkpoints.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        (validating_y_true_evolution[epoch], 
        validating_y_pred_evolution[epoch], 
        validating_g_evolution[epoch], 
        validating_idx_evolution[epoch]) = dataset_inference(model, validating_dataloader)
                
    torch.save(validating_y_pred_evolution, f'{output_dir}/validating_y_pred_evolution.pt')
    torch.save(validating_y_true_evolution, f'{output_dir}/validating_y_true_evolution.pt')
    torch.save(validating_g_evolution, f'{output_dir}/validating_g_evolution.pt')
    torch.save(validating_idx_evolution, f'{output_dir}/validating_idx_evolution.pt')

    validating_y_pred_evolution = torch.load(f'{output_dir}/validating_y_pred_evolution.pt', map_location=device)
    validating_y_true_evolution = torch.load(f'{output_dir}/validating_y_true_evolution.pt', map_location=device)
    validating_g_evolution = torch.load(f'{output_dir}/validating_g_evolution.pt', map_location=device)

    group_accuracy_dict = {0:[], 1:[], 2:[], 3:[]}

    for i in range(20):
        for j in range(len(validating_set_labels)):
            group_mask = validating_g_evolution[i]==j
            y_true_by_instance_group = validating_y_true_evolution[i, group_mask].argmax(axis=-1)
            y_pred_by_instance_group = validating_y_pred_evolution[i, group_mask].argmax(axis=-1)

            accuracy = accuracy_score(y_true_by_instance_group.cpu(), y_pred_by_instance_group.cpu())
            group_accuracy_dict[j].append(accuracy)
            
    plt.figure(figsize=(15,5))
    full_group_accuracy_string = ''
    for i in range(len(validating_set_labels)):
        plt.plot(group_accuracy_dict[i], label=validating_set_labels[i])
        group_accuracy_string = f'group{i}'
        for j in range(len(group_accuracy_dict[i])):
            group_accuracy_string += f', {group_accuracy_dict[i][j]:.4f}'
        full_group_accuracy_string += group_accuracy_string+'\n'

    group_accuracy_array = np.array([group_accuracy_dict[i] for i in range(4)])
    average_group_accuracy = (ratio['training'][0]*group_accuracy_array[0]+ratio['training'][1]*group_accuracy_array[1]+ratio['training'][2]*group_accuracy_array[2]+ratio['training'][3]*group_accuracy_array[3])/ratio['training'].sum()
    full_group_accuracy_string += 'averag, ' + ', '.join([f'{x:.4f}' for x in average_group_accuracy]) + '\n'
        
    print(full_group_accuracy_string)
    with open(f'{output_dir}/group_accuracy_evolution_in_validating_set.txt', 'w') as file:
        file.write(full_group_accuracy_string)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_accuracy_evolution_in_validating_set.png', dpi=300)
    plt.savefig(f'{output_dir}/group_accuracy_evolution_in_validating_set.pdf', dpi=300)


if __name__ == '__main__':
    main()