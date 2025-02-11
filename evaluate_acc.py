import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
import pandas as pd
from datetime import datetime
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import models

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
from core.utils import print_training_info, StdRedirect

model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, metavar='N',
                    help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, metavar='N',
                    help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'cinic10'])

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-mode', type=str, choices=['random', 'coreset', 'stratified', 'swav', 'badge'])

parser.add_argument('--data-score-path', type=str)
parser.add_argument('--coreset-key', type=str)
parser.add_argument('--data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--class-balanced', type=int, default=0,
                    help='Set 1 to use the same class ratio as to the whole dataset.')
parser.add_argument('--coreset-ratio', type=float)

#### Double-end Pruning Setting ####
parser.add_argument('--mis-key', type=str)
parser.add_argument('--mis-data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--mis-ratio', type=float)

#### Reversed Sampling Setting ####
parser.add_argument('--reversed-ratio', type=float,
                    help="Ratio for the coreset, not the whole dataset.")

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

################### Load Pseudo Labels from DL models ###################
parser.add_argument('--load_pseudo', action='store_true', default=False)
parser.add_argument('--pseudo_train_label_path', type=str, help='Path for the pseudo train labels')
parser.add_argument('--pseudo_test_label_path', type=str, help='Path for the pseudo test')

######################### Setting for Future Use #########################
# parser.add_argument('--ckpt-name', type=str, default='model.ckpt',
#                     help='The name of the checkpoint.')
# parser.add_argument('--lr-scheduler', choices=['step', 'cosine'])
# parser.add_argument('--network', choices=model_names, default='resnet18')
# parser.add_argument('--pretrained', action='store_true')
# parser.add_argument('--augment', choices=['cifar10', 'rand'], default='cifar10')

args = parser.parse_args()
start_time = datetime.now()

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"


print(f'Dataset: {args.dataset}')
######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
os.makedirs(task_dir, exist_ok=True)
last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')

######################### Print setting #########################
sys.stdout=StdRedirect(log_path)
print_training_info(args, all=True)
#########################
print(f'Last ckpt path: {last_ckpt_path}')

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.join(args.data_dir, args.dataset)
print(f'Data dir: {data_dir}')

if args.dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir)
    if args.load_pseudo:
        #--pseudo_train_label_path example: ../datasets/cifar-10-pseudo2/training_batch_labels.txt
        print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
        trainset = CIFARDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
elif args.dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir)
    if args.load_pseudo:
        #--pseudo_train_label_path example: ../datasets/cifar-100-python/label.pt 
        print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
        trainset = CIFARDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
    print(f"length of train set - {len(trainset)}")
elif args.dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir)

######################### Coreset Selection #########################
coreset_key = args.coreset_key
coreset_ratio = args.coreset_ratio
coreset_descending = (args.data_score_descending == 1)
total_num = len(trainset)


if args.coreset:
    if args.coreset_mode not in ['random', 'swav', 'badge']:
        print(args.coreset_mode)
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)

    if args.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(trainset), num=args.coreset_ratio * len(trainset))

    if args.coreset_mode == 'coreset':
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key, ratio=args.coreset_ratio, descending=(args.data_score_descending == 1), class_balanced=(args.class_balanced == 1))

    if args.coreset_mode == 'stratified':
        mis_num = int(args.mis_ratio * total_num)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)

        coreset_num = int(args.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, coreset_key=args.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]
        print(f'Length of coreset: {len(coreset_index)}')

    if args.coreset_mode == 'swav':
        df = pd.read_csv(args.data_score_path)
        data_score = {
            'targets': torch.tensor(df['Image Index'].values),
            'score': torch.tensor(df['Difficulty Score'].values)
        }
        
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key, ratio=args.coreset_ratio, descending=(args.data_score_descending == 1), class_balanced=(args.class_balanced == 1))

    if args.coreset_mode == 'badge':
    # Load indices from badge_selected_indices.json
        with open('../distil/badge_selected_indices_1000.json', 'r') as f:
            badge_data = [json.loads(line) for line in f.readlines()]

        badge_data_sorted = sorted(badge_data, key=lambda x: x["round"])

        aggregated_indices = []

        total_num_indices_needed = int(len(trainset) * coreset_ratio)

        for round_data in badge_data_sorted[1:]:
            if len(aggregated_indices) >= total_num_indices_needed:
                break
            aggregated_indices.extend(round_data["indices"])

        if len(aggregated_indices) > total_num_indices_needed:
            aggregated_indices = aggregated_indices[:total_num_indices_needed]

        aggregated_indices = [int(idx) for idx in aggregated_indices]
        coreset_index = aggregated_indices
        
        print(f'Selected {len(trainset)} examples using badge coreset starting from round 1.')
    trainset = torch.utils.data.Subset(trainset, coreset_index)


######################### Coreset Selection end #########################

# load from best_ckpt_path
    

if args.dataset == 'cifar10':
    testset = CIFARDataset.get_cifar10_test(data_dir)
    if args.load_pseudo:
        print(f"Loading Pseudo dataset labels from {args.pseudo_test_label_path}")
        testset = CIFARDataset.load_custom_labels(testset, args.pseudo_test_label_path)
elif args.dataset == 'cifar100':
    testset = CIFARDataset.get_cifar100_test(data_dir)
    if args.load_pseudo:
        print(f"Loading Pseudo dataset labels from {args.pseudo_test_label_path}")
        testset = CIFARDataset.load_custom_labels(testset, args.pseudo_test_label_path)
elif args.dataset == 'svhn':
    testset = SVHNDataset.get_svhn_test(data_dir)
elif args.dataset == 'cinic10':
    testset = CINIC10Dataset.get_cinic10_test(data_dir)

print(f"length of test set - {len(testset)}")


testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=True, num_workers=16)


if args.dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
else:
    num_classes=100

if args.network == 'resnet18':
    print('resnet18')
    model = resnet('resnet18', num_classes=num_classes, device=device)
if args.network == 'resnet50':
    print('resnet50')
    model = resnet('resnet50', num_classes=num_classes, device=device)

if os.path.exists(best_ckpt_path):
    print(f'Loading model from {best_ckpt_path}')
    state = torch.load(best_ckpt_path)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    print(f'Loaded model from {best_ckpt_path}')
else: # throw error
    print(f'No model found at {best_ckpt_path}')
    exit()

# Print out evaluation only
print(f'Evaluate the model')
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test ACC: {100 * correct / total}%')