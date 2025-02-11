import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import os, pickle
from core.data import CustomImageNetDataset, IndexDataset, ImageNetDataset, CIFARDataset
from core.training import Trainer

import argparse


def evaluate_model(model, dataloader, criterion, device, name="Test"):
    trainer = Trainer()
    test_loss, test_acc = trainer.test(model, dataloader, criterion, device, log_interval=200, printlog=True, topk=1)
    print(f'{name} Loss: {test_loss:.4f} | {name} Accuracy: {test_acc*100:.2f}%')
    return test_acc

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluation')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--data-dir', type=str, default='../datasets/')
parser.add_argument('--base-dir', type=str, default='./data-model/imagenet')
parser.add_argument('--task-name', type=str, default='tmp')
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--load_pseudo', action='store_true', default=False)
parser.add_argument('--pseudo_train_label_path', type=str)
parser.add_argument('--pseudo_test_label_path', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task_dir = os.path.join(args.base_dir, args.task_name)
best_ckpt_path = os.path.join(task_dir, 'ckpt-best.pt')

if args.dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(args.data_dir)
    testset = CIFARDataset.get_cifar100_test(args.data_dir)

elif args.dataset == 'imagenet':
    trainset = CustomImageNetDataset(os.path.join(args.data_dir, 'train'))
    testset = CustomImageNetDataset(os.path.join(args.data_dir, 'val'), is_test=True)

# trainset = ImageNetDataset.get_ImageNet_train(os.path.join(args.data_dir, 'train'))
# testset = ImageNetDataset.get_ImageNet_test(os.path.join(args.data_dir, 'val'))

trainset = IndexDataset(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)

# import ipdb; ipdb.set_trace()
# for batch_idx, (idx, (inputs, targets)) in enumerate(trainloader):
#     inputs, targets = inputs.to(device), targets.to(device)


# for batch_idx, (inputs, targets) in enumerate(testloader):
#     inputs, targets = inputs.to(device), targets.to(device)
        


if os.path.exists(best_ckpt_path):
    print(f"Loading model from '{best_ckpt_path}'")
    checkpoint = torch.load(best_ckpt_path, map_location=device)


    
    if args.dataset == 'imagenet':
        model = torchvision.models.resnet34(pretrained=False)
    else:
        from core.model_generator import resnet
        model = resnet('resnet18', num_classes=100, device=device)


    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}

    model.load_state_dict(new_state_dict)

    model = torch.nn.parallel.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()

    print("Evaluating with ground truth labels...")
    ground_truth_acc = evaluate_model(model, testloader, criterion, device, name="Ground Truth")

    print(f"Ground Truth Accuracy: {ground_truth_acc*100:.2f}%")
    print('\n\n\n')

    if args.load_pseudo:
        print(f'Loading pseudo test labels from {args.pseudo_test_label_path}')

        if args.dataset == 'imagenet':
            train_labels = torch.load(args.pseudo_train_label_path)
            test_labels = torch.load(args.pseudo_test_label_path)
            trainset = CustomImageNetDataset(path=os.path.join(args.data_dir, 'train'),
                                            pseudo_labels=train_labels)
            testset = CustomImageNetDataset(path=os.path.join(args.data_dir, 'val'),
                                            pseudo_labels=test_labels,
                                            is_test=True
            )
        elif args.dataset == 'cifar100':
            trainset = CIFARDataset.get_cifar100_train(args.data_dir)
            trainset = CIFARDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
            testset = CIFARDataset.load_custom_labels(testset, args.pseudo_test_label_path)


        
        trainset = IndexDataset(trainset)


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)

        # import ipdb ; ipdb.set_trace()

        # # for batch_idx, (idx, (inputs, targets)) in enumerate(trainloader):
        # #     inputs, targets = inputs.to(device), targets.to(device)

        # for batch_idx, (inputs, targets) in enumerate(testloader):
        #     inputs, targets = inputs.to(device), targets.to(device)

        print("Evaluating with pseudo labels...")
        pseudo_acc = evaluate_model(model, testloader, criterion, device, name="Pseudo Labels")

        
        print(f"Pseudo Label Accuracy: {pseudo_acc*100:.2f}%")
else:
    print(f"No checkpoint found at '{best_ckpt_path}', cannot proceed with evaluation.")

## usage: python evaluate_help.py --data-dir ../data/imagenet --base-dir ./data-model/imagenet-new --task-name clip_pseudo --gpuid 0,1 --load_pseudo --pseudo_train_label_path score_and_label/imagenet_clip_label.pt --pseudo_test_label_path score_and_label/imagenet_clip_label-test.pt


## usage: python evaluate_help.py --data-dir ../data/imagenet --base-dir data-model/new-imagenet-pseudo-label-search/ --task-name budget-0.1-0.5 --gpuid 0,1 --load_pseudo --pseudo_train_label_path score_and_label/imagenet_clip_label.pt --pseudo_test_label_path score_and_label/imagenet_clip_label-test.pt

## usage: python evaluate_help.py --data-dir ../datasets --dataset cifar100 --base-dir ./data-model/cifar100/ccs-beta-search --task-name ccs-0.3-0.3 --gpuid 0,1 --load_pseudo --pseudo_train_label_path score_and_label/cifar100_clip_label.pt --pseudo_test_label_path score_and_label/cifar100_clip_label-test.pt

