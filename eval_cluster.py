import glob
from functools import partial
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from eval_cluster_utils import *

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F


def calculate_hungarian_misclassification_rate(pseudo_labels, labels):
    if not isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = torch.tensor(pseudo_labels)

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    max_label = max(labels.max().item(), pseudo_labels.max().item()) + 1  
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for p, l in zip(pseudo_labels, labels):
        confusion_matrix[l.item(), p.item()] += 1

    row_indices, col_indices = linear_sum_assignment(confusion_matrix, maximize=True)

    correct_predictions = confusion_matrix[row_indices, col_indices].sum()

    total_predictions = len(labels)
    misclassification_rate = (total_predictions - correct_predictions) / total_predictions

    return misclassification_rate



def plot_scatter(x_axis, values, outdir, fname, xlab="epoch",
                 ylab="AUROC", title="OOD AUROC & score CIFAR100 -> CIFAR10"):
    plt.figure(figsize=(10,10))
    plt.plot(x_axis, values, "-o")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(Path(outdir) / fname)


def _eval_setting_to_str(s):
    if not isinstance(s, tuple):
        return str(s)
    s = [str(x) for x in s]
    return '-'.join(s)


def print_results(d):
    for k, d_inner in d.items():
        print(k)
        for k_inner, v in d_inner.items():
            s = f'{_eval_setting_to_str(k_inner)}:'
            print(f'\t{s:<22} {v[-1]:.2f}')


def load_tensorboard_loss(path):
    tag = 'Train loss epoch'
    event_acc = EventAccumulator(str(next(path.glob('event*'))))
    event_acc.Reload()
    if tag in event_acc.Tags()['scalars']:
        return pd.DataFrame([{'Epoch': ev.step, 'loss': ev.value}
                             for ev in event_acc.Scalars(tag)]).set_index('Epoch')
    # Multihead case
    dfs = []
    for p in path.rglob('Train loss*/event*'):
        event_acc = EventAccumulator(str(p))
        event_acc.Reload()
        dfs.append(pd.DataFrame([{'Epoch': ev.step, 'loss': ev.value}
                                 for ev in event_acc.Scalars(tag)]).set_index('Epoch'))
    df = pd.concat(dfs)
    return df.groupby('Epoch').min()


def main():
    args = get_eval_args()
    cudnn.deterministic = True

    auroc_results = defaultdict(partial(defaultdict, list))
    cluster_results = {"cluster_acc": [], "nmi": [], "anmi": [], "ari": [],
                       "cluster_acc-train": [], "nmi-train": [], "anmi-train": [], "ari-train": []}
    loss_results = {"train_loss": []}

    checkpoint_list = glob.glob(os.path.join(args.ckpt_folder, "*.pth"))
    outdir = Path(args.ckpt_folder).expanduser().resolve()

    # Read hparams
    with open(outdir / 'hp.json', 'r') as f:
        hparams = json.load(f)
    if not args.ignore_hp_file:
        args.__dict__.update({k: v for k, v in hparams.items() if v is not None})

    # Load loss history
    losses_df = load_tensorboard_loss(outdir)

    # replace last saved checkpoint name to be last
    checkpoint_list = list(map(lambda st: str.replace(st, "checkpoint.pth", "checkpoint9999.pth"), checkpoint_list))
    checkpoint_list = sorted(checkpoint_list)
    checkpoint_list = list(map(lambda st: str.replace(st, "checkpoint9999.pth", "checkpoint.pth", ), checkpoint_list))
    epochs = []

    print(f"dataset: {args.dataset} \n Checkpoints found {len(checkpoint_list)}  \n {checkpoint_list} ")
    assert len(checkpoint_list) >= 1
    args.datapath = './data' if  args.dataset in ["CIFAR10", "CIFAR100", "STL10", "CIFAR20"] else args.datapath
    extractor = None
    best_ckpt = None
    best_cluster_acc = -1
    for ckpt in checkpoint_list[:-1]:
        print(ckpt)
        # Epoch number for next epoch is saved in the checkpoint
        epoch = torch.load(ckpt, map_location='cpu')['epoch'] - 1
        epochs.append(epoch)
        if extractor is None or args.no_cache:
            extractor = FeatureExtractionPipeline(args, cache_backbone=not args.no_cache, datapath=args.datapath)
        train_features, test_features, train_labels, val_labels = \
            extractor.get_features(ckpt)
            

        # Cluster performance test
        ( _ , max_indices) = torch.max(test_features, dim=1)
        max_indices = max_indices.cpu().numpy()
        cluster_acc, nmi, anmi, ari = utils.compute_metrics(val_labels, max_indices, min_samples_per_class=5)

        cluster_results["cluster_acc"].append(cluster_acc)
        cluster_results["nmi"].append(nmi)
        cluster_results["anmi"].append(anmi)
        cluster_results["ari"].append(ari)

        # Track best model based on clustering accuracy
        if cluster_acc > best_cluster_acc:
            best_cluster_acc = cluster_acc
            best_ckpt = ckpt

        # Cluster performance train
        ( _ , max_indices) = torch.max(train_features, dim=1)
        max_indices = max_indices.cpu().numpy()
        cluster_acc, nmi, anmi, ari = utils.compute_metrics(train_labels, max_indices, min_samples_per_class=5)

        cluster_results["cluster_acc-train"].append(cluster_acc)
        cluster_results["nmi-train"].append(nmi)
        cluster_results["anmi-train"].append(anmi)
        cluster_results["ari-train"].append(ari)

        # Loss
        if epoch in losses_df.index:
            train_loss = losses_df.loc[epoch].item()
            loss_results["train_loss"].append(train_loss)
        else:
            loss_results["train_loss"].append(np.nan)

        print('\n', '-'*100, '\n')

        dict_data = {
                    "cluster_val_acc" : np.max(cluster_results["cluster_acc"]),
                    "NMI" : np.max(cluster_results["nmi"]),
                    "ARI" : np.max(cluster_results["ari"]),
                    "ckpt-best-cluster-acc": checkpoint_list[np.argmax(cluster_results["cluster_acc"])],
                    }
        cluster_results.update(loss_results)
        df = pd.DataFrame(cluster_results, index=epochs)
        df.index.name = "Epoch"
        print(df[["cluster_acc",
                "nmi",
                "ari"
                ]])
    with open(outdir / "best-results.json", 'w') as f:
        json.dump(dict_data, f, indent=4)
    
    df.to_csv(outdir / "checkpoint_metrics.csv")

    # Use best model to generate pseudo-labels
    print(f"Using best model checkpoint '{best_ckpt}' to generate pseudo-labels...")
    train_features, test_features, train_labels, val_labels = extractor.get_features(best_ckpt)

    # Generate and save labels
    pseudo_train_labels = torch.argmax(train_features, dim=1).cpu().numpy()
    pseudo_test_labels = torch.argmax(test_features, dim=1).cpu().numpy()

    torch.save(pseudo_train_labels, outdir / 'pseudo_label.pt')
    train_missclassification_rate = calculate_hungarian_misclassification_rate(pseudo_train_labels, train_labels)
    print(f"Train missclassification rate: {train_missclassification_rate:.4f}")
    torch.save(pseudo_test_labels, outdir / 'pseudo_label-test.pt')
    test_missclassification_rate = calculate_hungarian_misclassification_rate(pseudo_test_labels, val_labels)
    print(f"Test missclassification rate: {test_missclassification_rate:.4f}")


    print(f"Pseudo-labels saved for training and testing sets.")


if __name__ == '__main__':
    main()
