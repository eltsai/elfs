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
from core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset, STL10Dataset
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
parser.add_argument('--lr_policy', type=str, default='cosine', choices=['OneCycleLR', 'ReduceLROnPlateau', 'cosine', 'step'])
parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'cinic10', 'stl10'])

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')
parser.add_argument('--ignore-td', action='store_true', default=False)

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../datasets/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-mode', type=str, choices=['random', 'coreset', 'stratified', 'swav', 'badge', 'budget'])


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
parser.add_argument('--load-pseudo', action='store_true', default=False)
parser.add_argument('--pseudo-train-label-path', type=str, help='Path for the pseudo train labels')
parser.add_argument('--pseudo-test-label-path', type=str, help='Path for the pseudo test')

######################### Save Coreset Index for Plotting #########################
parser.add_argument('--save-coreset', action='store_true', default=True)
parser.add_argument('--end-early', action='store_true', default=False)

######################### Setting for Future Use #########################
parser.add_argument('--load-from-best', action='store_true', default=False)
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
trainset_label_path = os.path.join(task_dir, f'trainset-labels.pt')
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

valset = None

if args.dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir)
elif args.dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir)
    print(f"length of train set - {len(trainset)}")
elif args.dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir)
elif args.dataset == 'stl10':
    trainset = STL10Dataset.get_stl10_train(data_dir)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir)
    valset = CINIC10Dataset.get_cinic10_train(data_dir, is_val=True)



if args.load_pseudo:
    if "cifar" in args.dataset:
        #--pseudo_train_label_path example: ../datasets/cifar-100-python/label.pt 
        print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
        trainset = CIFARDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
    if "svhn" in args.dataset:
        print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
        trainset = SVHNDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
    if "stl10" in args.dataset:
        print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
        trainset = STL10Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
    if "cinic" in args.dataset:
        print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
        trainset = CINIC10Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
        valset = CINIC10Dataset.load_custom_labels(valset, args.pseudo_train_label_path, is_val=True)

if valset:
    # merge trainset and valset
    trainset = torch.utils.data.ConcatDataset([trainset, valset])


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

    if args.coreset_mode == 'budget':
        mis_num = int(args.mis_ratio * total_num)
        coreset_num = int(args.coreset_ratio * total_num)
        high_aum_chop_num = total_num - mis_num - coreset_num
        coreset_index = CoresetSelection.direct_selection(data_score, 
                                                          mis_key='accumulated_margin', 
                                                          mis_num=mis_num, 
                                                          mis_descending=False, 
                                                          coreset_key=args.coreset_key,
                                                          chop_num=high_aum_chop_num)

        print(f'Length of coreset: {len(coreset_index)}')

        

    if args.coreset_mode == 'swav':
        enhance = False
        # load pickle file
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)
        # data score: list of [index, distance, pseudo_label assigned by kmeans]
        # sort by distance: descending
        data_score = sorted(data_score, key=lambda x: x[1], reverse=True)
        print(f"Loaded data score from {args.data_score_path}")
        print(f'Length of data score: {len(data_score)}')
        # calculate number of coreset to select
        coreset_num = int(args.coreset_ratio * total_num)
        # select the first coreset_num indices
        if enhance:
            coreset_index = CoresetSelection.select_balanced_coreset_prototypicality(data_score, coreset_num)
        else:
            coreset_index = [x[0] for x in data_score[:coreset_num]]

    if args.coreset_mode == 'badge':
        with open(args.data_score_path, 'r') as f:
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
    
    if args.save_coreset:
        # save the coreset as .pt file -  set as default True
        coreset_index_path = os.path.join(task_dir, f'coreset_index.pt')
        with open(coreset_index_path, 'wb') as f:
            pickle.dump(coreset_index, f)
        print(f'Saved coreset index to {coreset_index_path}')
        if args.end_early:
            sys.exit(0)
    
    trainset = torch.utils.data.Subset(trainset, coreset_index)




######################### Coreset Selection end #########################


trainset = IndexDataset(trainset)
print(f"length of train set - {len(trainset)}")
data_dir = os.path.join(args.data_dir, args.dataset)
print("first 100 labels in trainset:")
print([int(trainset[i][1][1]) for i in range(100)])


if args.dataset == 'cifar10':
    testset = CIFARDataset.get_cifar10_test(data_dir)
elif args.dataset == 'cifar100':
    testset = CIFARDataset.get_cifar100_test(data_dir)
elif args.dataset == 'svhn':
    testset = SVHNDataset.get_svhn_test(data_dir)
elif args.dataset == 'stl10':
    testset = STL10Dataset.get_stl10_test(data_dir)
elif args.dataset == 'cinic10':
    testset = CINIC10Dataset.get_cinic10_test(data_dir)


if args.load_pseudo:
    if "cifar" in args.dataset:
        print(f"Loading Pseudo dataset labels from {args.pseudo_test_label_path}")
        testset = CIFARDataset.load_custom_labels(testset, args.pseudo_test_label_path)
    if "svhn" in args.dataset:
        print(f"Loading Pseudo dataset labels from {args.pseudo_test_label_path}")
        testset = SVHNDataset.load_custom_labels(testset, args.pseudo_test_label_path)
    if "stl10" in args.dataset:
        print(f"Loading Pseudo dataset labels from {args.pseudo_test_label_path}")
        testset = STL10Dataset.load_custom_labels(testset, args.pseudo_test_label_path)
    if "cinic" in args.dataset:
        print(f"Loading Pseudo dataset labels from {args.pseudo_test_label_path}")
        testset = CINIC10Dataset.load_custom_labels(testset, args.pseudo_test_label_path, is_test=True)

print(f"length of test set - {len(testset)}")
print('First 100 test label:')
print([int(testset[i][1]) for i in range(100)])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=True, num_workers=16)

# import ipdb ; ipdb.set_trace()

# for batch_idx, (idx, (inputs, targets)) in enumerate(trainloader):
#     inputs, targets = inputs.to(device), targets.to(device)

# for batch_idx, (inputs, targets) in enumerate(testloader):
#     inputs, targets = inputs.to(device), targets.to(device)

iterations_per_epoch = len(trainloader)
if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations


if args.dataset in ['cifar10', 'svhn', 'cinic10', 'stl10']:
    num_classes=10
elif args.dataset == 'cifar100':
    num_classes=100

if args.network == 'resnet18':
    print('resnet18')
    model = resnet('resnet18', num_classes=num_classes, device=device)
if args.network == 'resnet50':
    print('resnet50')
    model = resnet('resnet50', num_classes=num_classes, device=device)

model=torch.nn.parallel.DataParallel(model).cuda()


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

if args.lr_policy == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)
elif args.lr_policy == 'OneCycleLR':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=num_of_iterations)

epoch_per_testing = args.iterations_per_testing // iterations_per_epoch

print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

trainer = Trainer()
if args.ignore_td:
    TD_logger = None
    print('Ignore training dynamics info.')
else:
    TD_logger = TrainingDynamicsLogger()


current_epoch = 0
best_acc = 0
best_epoch = -1
# check if load from best
if args.load_from_best:
    print('Load from best ckpt')
    state = torch.load(best_ckpt_path)

    model.load_state_dict(state['model_state_dict'])
    current_epoch = state['epoch']
    # report best acc
    test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)
    best_acc = test_acc
    best_epoch = current_epoch
    print(f'Best acc: {test_acc * 100:.2f}')





while num_of_iterations > 0:
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    trainer.train(current_epoch, -1, model, trainloader, optimizer, criterion, scheduler, device, TD_logger=TD_logger, log_interval=60, printlog=True)

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0 or num_of_iterations == 0:
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)

    current_epoch += 1
    # scheduler.step()

# last ckpt testing
test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)
if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)
######################### Save #########################
state = {
    'model_state_dict': model.state_dict(),
    'epoch': current_epoch - 1
}
torch.save(state, last_ckpt_path)
if not args.ignore_td:
    TD_logger.save_training_dynamics(td_path, data_name=args.dataset)

print(f'Total time consumed: {(datetime.now() - start_time).total_seconds():.2f}')