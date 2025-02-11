import fnmatch
import inspect
import json
import sys
from argparse import Namespace
import argparse
from pathlib import Path
import os
import numpy as np
from torch import nn
from typing import Optional, Union

import requests
import clip
import timm
import torch
from torchvision import models as torchvision_models, transforms
from tqdm import tqdm
from ast import literal_eval


from .multi_head import MultiHeadClassifier

from timm.models.helpers import load_state_dict

_AVAILABLE_MODELS = (
    "dino_resnet50",
    "dino_vits16",
    "dino_vitb16",
    "timm_resnet50",
    "timm_vit_small_patch16_224",
    "timm_vit_base_patch16_224",
    "timm_vit_large_patch16_224",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "msn_vit_small",
    "msn_vit_base",
    "msn_vit_large",
    "mocov3_vit_small",
    "mocov3_vit_base",
    "clip_ViT-B/16",
    "clip_ViT-L/14",
    "clip_RN50",
    "mae_vit_base",
    "mae_vit_large",
    "mae_vit_huge",
)

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.from_numpy(train_labels).cuda()
        test_labels = torch.from_numpy(test_labels).cuda()

    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()

        temp = torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            )

        probs = torch.sum(temp,1)
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5



def available_models(pattern=None):
    if pattern is None:
        return _AVAILABLE_MODELS
    return tuple(fnmatch.filter(_AVAILABLE_MODELS, pattern))


def _load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)


def _download(url: str, filename: Path):
    """from https://stackoverflow.com/a/37573701"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception(f"Could not download from {url}")


_dict_models_urls = {
    "msn": {
        'vit_small_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar',
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vitb16_600ep.pth.tar',
        'vit_large_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vitl16_600ep.pth.tar',
        "key": 'target_encoder'
    },
    "mae": {
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        'vit_large_patch16_224': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        'vit_huge_patch14_224_in21k': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        "key": 'model'
    },

    "mocov3": {
        'vit_small_patch16_224': 'https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar',
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar',
        "key":  "state_dict"
    },
}

_dict_timm_names = {
        "vit_huge": 'vit_huge_patch14_224_in21k',
        "vit_large": 'vit_large_patch16_224',
        "vit_base": 'vit_base_patch16_224',
        "vit_small": 'vit_small_patch16_224',
        "vit_tiny": 'vit_tiny_patch16_224',
        "resnet50": 'resnet50',
        }


def _get_checkpoint_path(model_name: str, timm_base=True):
    if timm_base:
        name = _get_timm_name(model_name)
    prefix = model_name.split("_")[0]
    model_url = _dict_models_urls[prefix][name]
    print(f"Loading {model_url}")
    root = Path('~/.cache/torch/checkpoints').expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{model_name}.pth'
    if not path.is_file():
        print('Downloading checkpoint...')
        _download(model_url, path)
        d = torch.load(path, map_location='cpu')
        ckpt_key_name = _dict_models_urls[prefix]["key"]
        if ckpt_key_name in d.keys():
            state_dict = d[ckpt_key_name]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("momentum_encoder.", ""): v for k, v in state_dict.items()} # for mocov3
            torch.save(state_dict, path)
        else:
            raise KeyError(f"{ckpt_key_name} not found. Only {d.keys()} are available.")
    return path


def _get_timm_name(model_name: str):
    prefix = model_name.split("_")[0]
    # remove prefix
    model_name = model_name.replace("".join([prefix,"_"]),"")
    if model_name in _dict_timm_names.keys():
        return _dict_timm_names[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")


def build_arch(model_name: str):
    timm_name = _get_timm_name(model_name)
    model = timm.create_model(
        timm_name,
        in_chans=3,
        num_classes=0,
        pretrained=False)
    _load_checkpoint(model, _get_checkpoint_path(model_name), strict=False)
    return model


def kv_pair(s):
    # For extra arparse arguments
    k, v = s.split("=")
    try:
        # v is float/int etc, parse it
        v = literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    return k, v

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_args_parser():
    parser = argparse.ArgumentParser('MI clustering', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='clip_ViT-B/32', type=str,
                        help="""Name of architecture to train. For quick experiments with ViTs,
                        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--datapath', default='./data', type=str)
    parser.add_argument('--embed_dim', default=None, type=int)
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update.
        The value is increased to max_momentum_teacher during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--max_momentum_teacher', default=1.0, type=float)
    parser.add_argument('--use_bn_in_head', default=False, type=bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--head_dropout_prob', default=0.0, type=float,
                        help="Dropout probability in projection head (Default: 0.0)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.1, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.1, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.0001, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")

    bs_group = parser.add_mutually_exclusive_group()
    bs_group.add_argument('--batch_size_per_gpu', default=64, type=int,
                          help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    bs_group.add_argument('--batch_size', default=None, type=int,
                          help='Total batch size')

    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop and aug
    parser.add_argument('--vit_image_size', type=int, default=224, help="""image size that enters vit; 
        must match with patch_size: num_patches = (vit_image_size/patch_size)**2""")
    parser.add_argument('--image_size', type=int, default=32, help="""image size of in-distibution data. 
        negative samples are first resized to image_size and then inflated to vit_image_size. This
        ensures that aux samples have same resolution as in-dist samples""")
    parser.add_argument('--aug_image_size',type=int, default=None,
                       help='Image size for data augmentation. If None, use vit_image_size')

    parser.add_argument('--num_augs', type=int, default=1)

    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--knn_path', default=None, type=str)
    parser.add_argument('--dataset', default='CIFAR100',
                        choices=['CIFAR100', 'CIFAR10', "STL10", "CIFAR20", "IN1K", "IN50", 'IN100', "IN200"],
                        type=str)
    parser.add_argument('--knn', type=int, default=50, help='Number of nearest neighbors to use')

    parser.add_argument('--output_dir', default=None, type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--out_dim', default=1000, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=False, type=bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")

    parser.add_argument('--nlayers', default=2, type=int, help='Head layers')
    parser.add_argument('--hidden_dim', default=512, type=int, help="Head's hidden dim")
    parser.add_argument('--bottleneck_dim', default=256, type=int, help="Head's bottleneck dim")
    parser.add_argument('--l2_norm', default=False, action='store_true', help="Whether to apply L2 norm after backbone")
    parser.add_argument('--embed_norm', default=False, action='store_true',
                        help="Whether to normalize embeddings using precomputed mean and std")

    add_from_signature(parser, MultiHeadClassifier, "MultiHeadClassifier")

    parser.add_argument('--disable_ddp', default=False, action='store_true', help="Don't use DDP")

    precomputed_group = parser.add_mutually_exclusive_group()
    precomputed_group.add_argument('--train_backbone', default=False, action='store_true',
                        help="Don't share backbones between teacher and student")
    precomputed_group.add_argument('--precomputed', action='store_true', help="Use precomputed embeddings", default=False)

    parser.add_argument("--loss", default="TEMI", help="The name of one of the classes in losses",
                        choices=['WMI', 'PMI', 'TEMI', 'SCAN'])
    parser.add_argument("--loss-args", type=kv_pair, nargs="*", default={},
                        help="Extra arguments for the loss class")
    parser.add_argument("--loader", default='EmbedNN', help="The name of one of the classes in loaders")
    parser.add_argument("--loader-args", type=kv_pair, nargs="*", default={},
                        help="Extra arguments for the loader class")
    parser.add_argument('--new-run', action='store_true', help="Create a new directory for this run", default=False)

    return parser


def add_from_signature(parser, function, name=None):
    """Add arguments from a function signature to an existing parser."""
    if name is None:
        name = function.__name__
    group = parser.add_argument_group(name)
    signature = inspect.signature(function)
    for name, param in signature.parameters.items():
        default = param.default
        if param.kind == param.VAR_KEYWORD or default is param.empty:
            continue
        try:
            group.add_argument("--" + name, default=param.default, type=type(default))
        except argparse.ArgumentError:
            # Ignore arguments that are already added
            pass


def set_default_args(config):
    if hasattr(config, '__dict__'):
        config = vars(config)
    args = get_args_parser().parse_args([])
    args.__dict__.update(config)
    return args

# @torch.no_grad()
# def embed_dim(args, model):
#     from model_builders import load_embeds
#     try:
#         return load_embeds(args).shape[-1]
#     except Exception:
#         pass
#     if isinstance(model, nn.Module):
#         p = _backbone_param(model)
#         dummy_in = torch.empty(1, 3, args.vit_image_size, args.vit_image_size,
#                                device=p.device, dtype=p.dtype)
#         dummy_out = model(dummy_in)
#         return dummy_out.size(-1)
#     raise ValueError('Could not infer embed_dim')

def load_model(config, head=True, split_preprocess=False):
    """
    config/args file
    head=False returns just the backbone for baseline evaluation
    split_preprocess=True returns resizing etc. and normalization/ToTensor as separate transforms
    """
    
    config = set_default_args(config)

    preprocess = None

    if config.precomputed:
        backbone = config.arch
    elif "timm" in config.arch:  # timm models
        arch = config.arch.replace("timm_", "")
        arch = arch.replace("timm-", "")
        backbone = timm.create_model(arch, pretrained=True, in_chans=3, num_classes=0)
    elif "swav" in config.arch:
        backbone = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        backbone.head = None
    elif "swag" in config.arch:
        arch = config.arch.replace("swag_", "")
        backbone = torch.hub.load("facebookresearch/swag", model=arch)
        backbone.head = None
    elif "dino" in config.arch: 
        # dinov2_vitg14, dinov2_vitl14
        arch = config.arch.replace("-", "_")
        backbone = torch.hub.load('facebookresearch/dinov2', arch)
    elif "clip" in config.arch:
        arch = config.arch.replace("clip_", "")
        arch = arch.replace("clip-", "")
        assert arch in clip.available_models()
        clip_model, preprocess = clip.load(arch)
        backbone = clip_model.visual
    elif "mae" in config.arch or "msn" in config.arch or "mocov3" in config.arch:
        backbone = build_arch(config.arch)
    elif "convnext" in config.arch:
        backbone = getattr(torchvision_models, config.arch)(pretrained=True)
        backbone.classifier = torch.nn.Flatten(start_dim=1, end_dim=-1)

    elif config.arch in torchvision_models.__dict__.keys(): # torchvision models
        backbone = torchvision_models.__dict__[config.arch](num_classes=0)
    else:
        print(f"Architecture {config.arch} non supported")
        sys.exit(1)
    if not config.precomputed:
        print(f"Backbone {config.arch} loaded.")
    else:
        print("No backbone loaded, using precomputed embeddings from", config.arch)

    if preprocess is None:
        # imagenet means/stds
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        resize = transforms.Resize(config.vit_image_size, transforms.InterpolationMode.BICUBIC)
        if config.dataset=="IN1K":
            preprocess = transforms.Compose([
                transforms.Resize(int(256 * config.vit_image_size / 224)),
                transforms.CenterCrop(config.vit_image_size)
            ])
        else:
            preprocess = resize
        if not split_preprocess:
            preprocess = transforms.Compose([
                preprocess,
                transforms.ToTensor(),
                normalize
            ])
    elif split_preprocess:
        preprocess_aux = preprocess.transforms[:-2]
        normalize_aux = preprocess.transforms[-2:]
        preprocess = transforms.Compose(preprocess_aux)
        normalize = transforms.Compose(normalize_aux)

    model = backbone.float()

    if split_preprocess:
        return model, preprocess, normalize 
    print(preprocess)   
    return model, preprocess


def _build_from_config(
        precomputed: bool,
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    if isinstance(config, str) or isinstance(config, Path):
        p = Path(config)
        with open(p, "r") as f:
            config = json.load(f)
        config = Namespace(**config)
    if config is None:
        config = Namespace()
    config.num_heads = 1
    config.precomputed = precomputed
    if ckpt_path is not None:
        # Don't reload norms
        config.embed_norm = False

    d = None
    if ckpt_path is not None:
        d = torch.load(ckpt_path, map_location="cpu")
        if 'teacher' in d:
            d = d['teacher']
        if 'head.best_head_idx' in d:
            best_head_idx = d['head.best_head_idx']
            d2 = {k: v for k, v in d.items() if k in ('embed_mean', 'embed_std')}
            d2['head.best_head_idx'] = torch.tensor(0)
            for k, v in d.items():
                if k.startswith(f'head.heads.{best_head_idx}.'):
                    k = 'head.heads.0.' + k[len(f'head.heads.{best_head_idx}.'):]
                    d2[k] = v
            d = d2
        else:
            d['head.best_head_idx'] = torch.tensor(0)
        config.embed_dim = d['head.heads.0.mlp.0.weight'].size(1)

    model, _ = load_model(config, head=True)
    model.eval()
    if d is not None:
        model.load_state_dict(d, strict=False)

    return model


def build_head_from_config(
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    """
    config: Either path to hp.json or config namespace
    ckpt_path: Path to checkpoint
    """
    return _build_from_config(True, config, ckpt_path)


def build_model_from_config(
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    """
    config: Either path to hp.json or config namespace
    ckpt_path: Path to checkpoint
    """
    return _build_from_config(False, config, ckpt_path)


def load_embeds(config=None,
                arch=None,
                dataset=None,
                test=False,
                norm=False,
                datapath='data',
                with_label=False):
    p, test_str = _embedding_path(arch, config, datapath, dataset, test)
    emb = torch.load(p / f'embeddings{test_str}.pt', map_location='cpu')
    if norm:
        emb /= emb.norm(dim=-1, keepdim=True)
    if not with_label:
        return emb
    label = torch.load(p / f'label{test_str}.pt', map_location='cpu')
    return emb, label


def _embedding_path(arch, config, datapath, dataset, test):
    assert bool(config) ^ bool(arch and dataset)
    if config:
        arch = config.arch
        dataset = config.dataset
    import gen_pseudo_labels
    test_str = '-test' if test else ''
    p = gen_pseudo_labels.get_outpath(arch, dataset, datapath)
    return p, test_str


def load_embed_stats(
        config=None,
        arch=None,
        dset=None,
        test=False,
        datapath='data'):
    p, test_str = _embedding_path(arch, config, datapath, dset, test)
    mean = torch.load(p / f'mean{test_str}.pt', map_location='cpu')
    std = torch.load(p / f'std{test_str}.pt', map_location='cpu')
    return mean, std



import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.datasets as tds
from pathlib import Path


def load_embeds(config=None,
                arch=None,
                dataset=None,
                test=False,
                norm=False,
                datapath='data',
                with_label=False):
    p, test_str = _embedding_path(arch, config, datapath, dataset, test)
    emb = torch.load(p / f'embeddings{test_str}.pt', map_location='cpu')
    if norm:
        emb /= emb.norm(dim=-1, keepdim=True)
    if not with_label:
        return emb
    label = torch.load(p / f'label{test_str}.pt', map_location='cpu')
    return emb, label

class PrecomputedEmbeddingDataset(Dataset):

    def __init__(self, dataset, arch, train, datapath):
        super().__init__()
        self.emb, self.targets = load_embeds(
            arch=arch,
            dataset=dataset,
            datapath=datapath,
            with_label=True,
            test=not train)

    def __getitem__(self, index):
        return self.emb[index], self.targets[index]

    def __len__(self):
        return len(self.emb)


def get_dataset(dataset, datapath='./data', train=True, transform=None, download=True, precompute_arch=None):
    if precompute_arch:
        return PrecomputedEmbeddingDataset(
            dataset=dataset,
            arch=precompute_arch,
            datapath="data", # assumes embeddings are saved in the ./data folder
            train=train)
    
    load_obj = tds if dataset in ["CIFAR10","CIFAR100", "STL10"] else loaders
    if dataset == "STL10":
        split = 'train' if train else 'test'
        return getattr(load_obj, dataset)(root=datapath,
                        split=split,
                        download=download, transform=transform)
    elif "CIFAR" in dataset:
        return getattr(load_obj, dataset)(root=datapath,
                        train=train,
                        download=download, transform=transform)
    else:
        # imagenet subsets
        # TODO i dont know if val and val_structured are the same
        if "ILSVRC" in datapath and train is False:
            datapath = datapath.replace("train","val")
        return getattr(load_obj, dataset)(root=datapath,
                         transform=transform)

import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.datasets as tds
from pathlib import Path

class EmbedNN(Dataset):
    def __init__(self,
                 knn_path,
                 transform,
                 k=10,
                 dataset="CIFAR100",
                 datapath='./data',
                 precompute_arch=None):
        super().__init__()
        self.transform = transform
        self.complete_neighbors = torch.load(knn_path)
        if k < 0:
            k = self.complete_neighbors.size(1)
        self.k = k
        self.neighbors = self.complete_neighbors[:, :k]
        self.datapath = './data' if 'IN' not in dataset else datapath

        self.dataset = get_dataset(
            dataset,
            datapath=datapath,
            transform=None,
            train=True,
            download=True,
            precompute_arch=precompute_arch)

    def get_transformed_imgs(self, idx, *idcs):
        img, label = self.dataset[idx]
        rest_imgs = (self.dataset[i][0] for i in idcs)
        return self.transform(img, *rest_imgs), label

    def __getitem__(self, idx):
        # KNN pair
        pair_idx = np.random.choice(self.neighbors[idx], 1)[0]

        return self.get_transformed_imgs(idx, pair_idx)

    def __len__(self):
        return len(self.dataset)


class TruePosNN(EmbedNN):

    def __init__(self, knn_path, *args, **kwargs):
        super().__init__(knn_path, *args, **kwargs)
        p = Path(knn_path).parent
        nn_p = p / 'hard_pos_nn.pt'
        if nn_p.is_file():
            self.complete_neighbors = torch.load(nn_p)
        else:
            emb = torch.load(p / 'embeddings.pt')
            emb /= emb.norm(dim=-1, keepdim=True)
            d = emb @ emb.T
            labels = torch.tensor(self.dataset.targets)
            same_label = labels.view(1, -1) == labels.view(-1, 1)
            # Find minimum number of images per class
            k_max = same_label.sum(dim=1).min()
            d.fill_diagonal_(-2)
            d[torch.logical_not(same_label)] = -torch.inf
            self.complete_neighbors = d.topk(k_max, dim=-1)[1]
            torch.save(self.complete_neighbors, nn_p)
        self.neighbors = self.complete_neighbors[:, :self.k]


