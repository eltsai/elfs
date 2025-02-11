import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
from datetime import datetime

from torchvision import models

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import CoresetSelection, IndexDataset, CustomImageNetDataset
from core.utils import print_training_info, StdRedirect

model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, metavar='N',
                    help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, metavar='N',
                    help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet34'])
parser.add_argument('--scheduler', type=str, default='default', choices=['default', 'short', 'cosine', 'short-400k'])

parser.add_argument('--ignore-td', action='store_true', default=False)
parser.add_argument('--load-from-best', action='store_true', default=False,
                    help='If set, training starts from the best checkpoint')

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../datasets/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str, default='./data-model/imagenet',
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

################### Load Pseudo Labels from DL models ###################
parser.add_argument('--load-pseudo', action='store_true', default=False)
parser.add_argument('--pseudo-train-label-path', type=str, help='Path for the pseudo train labels')
parser.add_argument('--pseudo-test-label-path', type=str, help='Path for the pseudo test')

######################### Save Coreset Index for Plotting #########################
parser.add_argument('--save-coreset', action='store_true', default=True)
parser.add_argument('--end-early', action='store_true', default=False)

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-mode', type=str)

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

parser.add_argument('--strata', type=int, default=50)

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')
parser.add_argument('--local_rank', type=str)



args = parser.parse_args()
start_time = datetime.now()

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
os.makedirs(task_dir, exist_ok=True)
td_dir = os.path.join(task_dir, 'training-dynamics')
os.makedirs(td_dir, exist_ok=True)

last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')


######################### Print setting #########################
sys.stdout=StdRedirect(log_path)
print_training_info(args, all=True)
#########################
print(f'Last ckpt path: {last_ckpt_path}')
print(f'Training log path: {td_dir}')

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = args.data_dir
train_labels = None
if args.load_pseudo:
    print(f'Loading pseudo labels from {args.pseudo_train_label_path}')
    train_labels = torch.load(args.pseudo_train_label_path)
trainset = CustomImageNetDataset(path=os.path.join(args.data_dir, 'train'),
                                 pseudo_labels=train_labels)


######################### Coreset Selection #########################
coreset_key = args.coreset_key
coreset_ratio = args.coreset_ratio
coreset_descending = (args.data_score_descending == 1)
total_num = len(trainset)

if args.coreset:
    if args.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(trainset), num=args.coreset_ratio * len(trainset))
    else:
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)

    if args.coreset_mode == 'coreset':
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key, ratio=args.coreset_ratio, descending=(args.data_score_descending == 1), class_balanced=(args.class_balanced == 1))

    if args.coreset_mode == 'stratified':
        mis_num = int(args.mis_ratio * total_num)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)

        print(f'Strata: {args.strata}')
        coreset_num = int(args.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, coreset_key=args.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]

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
    if args.save_coreset:
        # save as .pt file
        coreset_index_path = os.path.join(task_dir, f'coreset_index.pt')
        with open(coreset_index_path, 'wb') as f:
            pickle.dump(coreset_index, f)
        print(f'Saved coreset index to {coreset_index_path}')
        if args.end_early:
            sys.exit(0)


    trainset = torch.utils.data.Subset(trainset, coreset_index)
    print(len(trainset))

    
######################### Coreset Selection end #########################

trainset = IndexDataset(trainset)
print(len(trainset))

print('First 100 train label:')
print([trainset[i][1][1] for i in range(100)])

test_labels = None
if args.load_pseudo:
    print(f'Loading pseudo labels from {args.pseudo_test_label_path}')
    test_labels = torch.load(args.pseudo_test_label_path)
testset = CustomImageNetDataset(path=os.path.join(args.data_dir, 'val'),
                                pseudo_labels=test_labels,
                                is_test=True)
print(len(testset))
print('First 100 test label:')
print([testset[i][1] for i in range(100)])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size * 2, shuffle=True, pin_memory=True, num_workers=8)

iterations_per_epoch = len(trainloader)
print(iterations_per_epoch)

if args.network == 'resnet34':
    print('Using resnet34.')
    model = torchvision.models.resnet34(pretrained=False, progress=True)
if args.network == 'resnet50':
    print('Using resnet50.')
    model = torchvision.models.resnet50(pretrained=False, progress=True)

model=torch.nn.parallel.DataParallel(model).cuda()

if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations

epoch_per_testing = max(args.iterations_per_testing // iterations_per_epoch, 1)

print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print(f'Using scheduler: {args.scheduler}!')
if args.scheduler == 'default':
    scheduler_epoch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,100], gamma=0.1)
    scheduler_iteration = None
elif args.scheduler == 'short': 
    # scheduler_epoch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,65], gamma=0.1)
    scheduler_epoch = None
    scheduler_iteration = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80000, 160000, 240000, 270000], gamma=0.1)
elif args.scheduler == 'short-400k': 
    # scheduler_epoch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,65], gamma=0.1)
    scheduler_epoch = None
    scheduler_iteration = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 200000, 300000, 350000], gamma=0.1)
elif args.scheduler == 'cosine': 
    scheduler_epoch = None
    scheduler_iteration = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)

trainer = Trainer()

if args.load_from_best:
    if os.path.exists(best_ckpt_path):
        print(f"Loading model and optimizer state from best checkpoint '{best_ckpt_path}'")
        checkpoint = torch.load(best_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        current_epoch = checkpoint.get('epoch', 0) + 1  # Proceed from the next epoch
        best_epoch = current_epoch - 1  # Assume the loaded checkpoint was the best
        best_acc = checkpoint.get('best_acc', 0)  # Load best accuracy if it was saved
    else:
        print(f"No checkpoint found at '{best_ckpt_path}', starting from scratch.")
        current_epoch = 0
        best_acc = 0
        best_epoch = -1
else:
    current_epoch = 0
    best_acc = 0
    best_epoch = -1

while num_of_iterations > 0:
    if args.ignore_td:
        TD_logger = None
        print('Ignore training dynamics info.')
    else:
        TD_logger = TrainingDynamicsLogger()
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    trainer.train(current_epoch, -1, model, trainloader, optimizer, criterion, scheduler_iteration, device, TD_logger=TD_logger, log_interval=1000, printlog=True)

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0:
        # test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=5)
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=1)

        if test_acc > best_acc:
            print('Updating best ckpt.')
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)

    current_epoch += 1

    if scheduler_epoch:
        scheduler_epoch.step()
        print(f'Current learing rate: {scheduler_epoch.get_last_lr()}.')
    else:
        print(f'Current learing rate: {scheduler_iteration.get_last_lr()}.')

    if not args.ignore_td:
        td_path = os.path.join(td_dir, f'td-{args.task_name}-epoch-{current_epoch}.pickle')
        print(f'Saving training dynamics at {td_path}')
        TD_logger.save_training_dynamics(td_path, data_name='imagenet')

print('Last ckpt evaluation.')
# test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=5)
test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=200,  printlog=True, topk=1)

print('done')
print(f'Total time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)

state = {
    'model_state_dict': model.state_dict(),
    'epoch': current_epoch - 1
}
torch.save(state, last_ckpt_path)