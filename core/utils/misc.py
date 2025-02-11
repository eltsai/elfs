import torch
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

def find_centroid_kmeans(embeddings, num_classes):
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centroids, labels

def calculate_distances(embeddings, labels, centroids):
    distances = []
    for i, embedding in enumerate(embeddings):
        centroid = centroids[labels[i]]
        distance = np.linalg.norm(embedding - centroid)
        distances.append([i, distance, labels[i]])
    return distances

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

def map_pseudo_label_hungarian(pseudo_labels, labels):
    """
    Maps pseudo labels to labels using the Hungarian algorithm and returns the mapped pseudo labels.

    Parameters:
    - pseudo_labels: Array of pseudo labels predicted by the model.
    - labels: Array of correct labels.

    Returns:
    - mapped_pseudo_labels: Array of pseudo labels mapped to the correct labels.
    """
    num_classes = len(np.unique(labels))

    # Create a confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pl, cl in zip(pseudo_labels, labels):
        confusion_matrix[pl, cl] += 1

    # Apply the Hungarian algorithm to minimize the cost of mismatches
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix) # Negate for maximization

    # Create a mapping from pseudo labels to labels
    mapping = dict(zip(row_ind, col_ind))

    # Map pseudo labels to their corresponding correct labels
    mapped_pseudo_labels = np.array([mapping[pl] for pl in pseudo_labels])

    return mapped_pseudo_labels



def compute_stats(outpath):
    for test in True, False:
        test_str = '-test' if test else ''
        embeddings = torch.load(outpath / f'embeddings{test_str}.pt', map_location='cpu')
        torch.save(embeddings.mean(dim=0), outpath / f'mean{test_str}.pt')
        torch.save(embeddings.std(dim=0), outpath / f'std{test_str}.pt')

def prediction_correct(true, preds):
    """
    Computes prediction_hit.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Prediction_hit for each img.
    """
    rst = (torch.softmax(preds, dim=1).argmax(dim=1) == true)
    return rst.detach().cpu().type(torch.int)

def get_model_directory(base_dir, model_name):
    model_dir = os.join(base_dir, model_name)
    ckpt_dir = os.join(model_dir, 'ckpt')
    data_dir = os.join(model_dir, 'data')
    log_dir = os.join(model_dir, 'log')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return ckpt_dir, data_dir, log_dir

def l2_distance(tensor1, tensor2):
    dist = (tensor1 - tensor2).pow(2).sum().sqrt()
    return dist

def accuracy(output, target, topk=1):
    """Computes the precision@k for the specified values of k"""
    maxk = topk
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct.float().sum()
    acc = correct_k * (100.0 / batch_size)

    return acc, correct_k.item()