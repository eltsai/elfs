# Effective Label-Free Selection (ELFS)


Code implementation for ICLR 2025 paper: [ELFS: Label-Free Coreset Selection with Proxy Training Dynamics](https://openreview.net/forum?id=yklJpvB7Dq)

ELFS is a label-free coreset selection method that employs deep clustering to estimate data difficulty scores without ground-truth labels. If you find this work useful in your research, please consider citing:

```
@inproceedings{zheng2025elfs,
  title={ELFS: Label-Free Coreset Selection with Proxy Training Dynamics},
  author={Zheng, Haizhong and Tsai, Elisa and Lu, Yifu and Sun, Jiachen and Bartoldson, Brian R and Kailkhura, Bhavya and Prakash, Atul},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

# Usage and Examples

**Environment**
```
conda env create -f environment.yml
```

## Pseudo-Label Generation

**Generate Embeddings and NNs**

For example, to use [DINO](https://github.com/facebookresearch/dino) to generate embeddings and nearest neighbors for CIFAR10: 
```
python gen_embeds.py --arch dino_vitb16 --dataset CIFAR10 --batch_size 256
```
Generated files will be stored under `data/embeddings/CIFAR10-dino_vitb16/` by default.

**Training Cluster Heads**

To train 50 cluster heads for 200 epochs with using the embeddings and 50-NNs:
```
export CUDA_VISIBLE_DEVICES=0; outdir=$"./experiments/cifar10-dino" ; clusters=10 ; dataset=$"CIFAR10"; 

python train_cluster_heads.py  --precomputed --arch dino_vitb16  --batch_size=1024 \
--use_fp16=true --max_momentum_teacher=0.996 --lr=1e-4 --warmup_epochs=20 --min_lr=1e-4 \
 --epochs=200 --output_dir $outdir --dataset $dataset  --knn=50 --out_dim=$clusters \
 --num_heads=50 --loss TEMI --loss-args  beta=0.6 --optimizer adamw
```

**Pseudo-Labels Generation**

```
python eval_cluster.py --ckpt_folder $outdir
```

The pseudo label generated for trainset and testset for CIFAR10 will be stored by default as `experiments/cifar10-dino/pseudo_label.pt` and `experiments/cifar10-dino/pseudo_label-test.pt`, respectively.


## Coreset Selection
**Training Dynamics Collection with Pseudo-Labels**

This step is **necessary** to collect training dynamics for future coreset selection.

**Important**: In the training dynamic collection, pseudo-labels need to be loaded, be sure to set `--load-pseudo` as `True` and load corresponding pseudo-labels.
```
python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 \
--network resnet18 --batch-size 256 --task-name all-data \
--base-dir ./data-model/cifar10 \
--load-pseudo --pseudo-train-label-path <path-to-cifar10_label.pt>  \
--pseudo-test-label-path <path-to-cifar10_label-test.pt> 
```


**Importance score calculation**
we need to calcualte the different importance scores (AUM, forgetting, etc) for coreset selection.

```
python generate_importance_score.py --dataset cifar10 --gpuid 0 \
--base-dir ./data-model/cifar10 --task-name all-data \
--load-pseudo --pseudo-train-label-path <path-to-cifar10_label.pt> 
```

The data score will be stored under `./data-model/cifar10/all-data/data-score-all-data.pickle` by default.

### Train a model with a specific coreset selection method:
Here we use 90% pruning rate on CIFAR10 as an example. At this stage, our coreset is annotated (ground-truth labels are now available). Use `--ignore-td` to avoid saving training dynamics.

**ELFS with AUM**
```
python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --task-name budget-0.1 \
--base-dir ./data-model/cifar10/ --coreset --coreset-mode budget \
--data-score-path <path-to-dino-pseudo-label-score> \
--coreset-key accumulated_margin \
--coreset-ratio 0.1 --mis-ratio 0.4 --ignore-td
```

**ELFS with forgetting**
```
python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --task-name budget-0.1-forgetting \
--base-dir ./data-model/cifar10/ --coreset --coreset-mode budget \
--data-score-path <path-to-dino-pseudo-label-score> --coreset-key forgetting --data-score-descending 1 \
--coreset-ratio 0.1 --mis-ratio 0.4 --ignore-td
```
You can also train with other sampling methods, such as random and el2n, see the [CCS code repo](https://github.com/haizhongzheng/Coverage-centric-coreset-selection) for more details.


# ImageNet training code
```
#Train imagenet classifier and collect the training dynamics

python train_imagenet.py --epochs 60 --lr 0.1 --scheduler cosine --task-name pseudo_dino \
--base-dir ./data-model/imagenet --data-dir <path-to-imagenet-data> --network resnet34 \
--batch-size 256 --gpuid 0,1 --load-pseudo \
--pseudo-train-label-path <path-to-imagenet-pseudo-labels.pt> \
--pseudo-test-label-path <path-to-imagenet-pseudo-labels-test.pt> 


#Calculate score for each example

python generate_importance_score_imagenet.py --data-dir <path-to-imagenet-data> \
--base-dir ./data-model/imagenet --task-name pseudo_dino --load_pseudo \
--pseudo-train-label-path <path-to-imagenet-pseudo-labels.pt> \
--data-score-path ./imagenet_dino_score.pt

#Train model with ELFS coreset 

#90% pruning rate
python train_imagenet.py --iterations 300000 --iterations-per-testing 5000 --lr 0.1 \
--scheduler cosine --task-name budget-0.1 --data-dir <path-to-imagenet-data> \
--base-dir ./data-model/imagenet --coreset --coreset-mode budget \
--data-score-path ./magenet_dino_score.pt --coreset-key accumulated_margin \
--network resnet34 --batch-size 256 --coreset-ratio 0.1 --mis-ratio 0.3 \
--data-score-descending 1 --gpuid 0,1 --ignore-td

```


# Acknowledgements
Thanks to the authors of [Exploring the Limits of Deep Image Clustering using Pretrained Models](https://github.com/HHU-MMBS/TEMI-official-BMVC2023) for releasing their code, our pseudo-label generation part is built upon their code.

Thanks to the authors of [Coverage-centric Coreset Selection for High Pruning Rates](https://github.com/haizhongzheng/Coverage-centric-coreset-selection). Much of this codebase is adapted from their code.

# Other Baselines

* Random Sampling
* [Active Learning (BADGE)](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.badge).
* Prototipicality ([SWaV](https://github.com/facebookresearch/swav)+k-means)
* [D2 Pruning](https://github.com/adymaharana/d2pruning/tree/master)
