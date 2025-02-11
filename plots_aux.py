import numpy as np
import matplotlib.pyplot as plt

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

import os
import numpy as np
import torch

from torchvision import datasets, transforms
import torchvision
from torch.utils.data import Subset

##############<-AUM->##############
import numpy as np
import matplotlib.pyplot as plt
from core.utils.misc import map_pseudo_label_hungarian


def plot_log_density_graph(gt_score, score, stride=2, num_ticks=5):
    """
    Plots the log density graph for the given ground truth and predicted scores.
    
    Parameters:
        gt_score (array-like): Ground truth scores.
        score (array-like): Predicted scores.
        stride (int): The stride for binning the data. Default is 2.
        num_ticks (int): Number of ticks to display on each axis. Default is 5.
    """
    # Creating bins and calculating density matrix
    bin_edges = np.arange(0, 204, stride) 
    density_matrix, x_edges, y_edges = np.histogram2d(gt_score, score, bins=[bin_edges, bin_edges])

    log_density_matrix = np.log(density_matrix + 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(log_density_matrix, cmap='hot_r', interpolation='nearest', origin='lower')
    plt.colorbar(label='Density')
    plt.title('Log Density Graph')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Scaling the tick labels
    original_range = np.arange(len(density_matrix))
    scaled_range = original_range * stride

    tick_indices = np.linspace(0, len(scaled_range)-1, num_ticks, dtype=int)  # Indices of ticks to label
    tick_labels = scaled_range[tick_indices]  # The labels to show, scaled

    plt.xticks(tick_indices, labels=tick_labels)
    plt.yticks(tick_indices, labels=tick_labels)

    plt.xlabel('Pseudo Label AUM')
    plt.ylabel('Ground Truth Label AUM')

    plt.show()



def plot_misclassification_rates(correct_labels, pseudo_labels, title, width=1, filename=None, figsize=(9.5, 7), color='blue'):
    """
    Plots the misclassification rates by class and the overall misclassification rate.
    
    Parameters:
    - correct_labels: Array of correct labels.
    - pseudo_labels: Array of pseudo labels predicted by the model.
    - title: String title for the plot.
    """
    pseudo_labels = map_pseudo_label_hungarian(pseudo_labels, correct_labels)
    misclassified = correct_labels != pseudo_labels
    misclassification_rate = np.mean(misclassified)

    print(f"Overall Misclassification Rate: {misclassification_rate * 100:.2f}%")

    num_classes = len(np.unique(correct_labels))
    misclassification_rates_by_class = []
    for class_id in range(num_classes):
        class_mask = correct_labels == class_id
        class_misclassified = misclassified[class_mask]
        class_misclassification_rate = np.mean(class_misclassified)
        misclassification_rates_by_class.append(class_misclassification_rate)

    plt.figure(figsize=figsize)
    bars = plt.bar(range(num_classes), misclassification_rates_by_class, label='Class Misclassification Rate', color=color, width=width)
    line = plt.axhline(y=misclassification_rate, color='r', linestyle='-', linewidth=2, label='Overall Misclassification Rate')

    # plt.xlabel('Class ID')
    # plt.ylabel('Misclassification Rate')
    # plt.title(title)
    # if num_classes == 10:
    #     plt.xticks(range(num_classes), ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], rotation=45)
    if num_classes == 100:
        # print number every 10 classes
        plt.xticks(range(0, num_classes, 10), range(0, num_classes, 10))
    # set x range from 0 to num_classes
    plt.xlim(0, num_classes)

    #plt.legend([bars, line], ['Class Misclassification Rate', f'Overall Misclassification Rate: {misclassification_rate * 100:.2f}%'])
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()



def calculate_stride(x_min, x_max):
    if x_max - x_min > 5000:
        stride = 1000
    elif x_max - x_min > 300:
        stride = 50
    elif x_max - x_min > 150:
        stride = 25
    elif x_max - x_min > 100:
        stride = 20
    else:
        stride = 10
    return stride


def plot_data_score_distribution_highlight_compare(score, title, first_index, first_name, second_index, second_name, bin_width=1.0, x_range=None, y_range=None, filename=None, color='#3a5bd4', background='all data'):
    """
    Plot the distribution of data scores, with scores at first_index highlighted in red
    and scores at second_index highlighted in green. 
    'bin_width' controls the granularity of the histogram bins.
    """
    
    _fontsize = 15
    title_fontsize = 17

    fig, ax = plt.subplots()
    ax.tick_params(axis="both", direction="in", length=2, width=1)

    x_min = score.min()
    x_max = score.max()
    bins = np.arange(x_min, x_max + bin_width, bin_width)

    n, x, _ = ax.hist(score, bins=bins, alpha=0.1, label=background, color=color)
    bin_centers = 0.5*(x[1:]+x[:-1])
    ax.plot(bin_centers, n, linewidth=1, c=color)  

    first_scores = score[first_index]
    ax.hist(first_scores, bins=x, alpha=0.5, label=first_name, color='red')

    second_scores = score[second_index]
    ax.hist(second_scores, bins=x, alpha=0.5, label=second_name, color='green', hatch='//')

    if x_range:
        ax.set_xlim(x_range)
    else:
        ax.set_xlim(x_min, x_max)
    if y_range:
        ax.set_ylim(y_range)

    ax.set_xlabel('Area under the margin (AUM)', fontsize=title_fontsize)
    ax.set_ylabel('#Examples', fontsize=title_fontsize)
    ax.legend(prop={'size': title_fontsize})
    ax.grid(zorder=-10, color='#eaeaf2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(fontsize=_fontsize)
    plt.yticks(fontsize=_fontsize)
    plt.rc('font', size=_fontsize)
    # plt.title(title, fontsize=title_fontsize)

    if filename:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()


def calculate_coverage_percentage(score, first_index, second_index, bin_width):
    """
    Calculate the percentage of the distribution of 'first_index' scores that is covered by 'second_index' scores.
    Coverage is defined as the sum of the minimum counts of 'first_index' and 'second_index' in each bin where both have entries.

    Parameters:
        score (np.array): Array of scores.
        first_index (list or np.array): Indices of the first group.
        second_index (list or np.array): Indices of the second group.
        bin_width (float): Width of each bin for histogram.

    Returns:
        float: Percentage of coverage.
    """
    
    min_score = min(score)
    max_score = max(score)
    bins = np.arange(min_score, max_score + bin_width, bin_width)

    first_hist, _ = np.histogram(score[first_index], bins=bins)
    second_hist, _ = np.histogram(score[second_index], bins=bins)

    min_overlap = np.minimum(first_hist, second_hist)
    covered_count = np.sum(min_overlap)
    total_first_count = np.sum(first_hist)  

    if total_first_count > 0:
        coverage_percentage = (covered_count / total_first_count) * 100
    else:
        coverage_percentage = 0 

    return coverage_percentage


def plot_data_score_distribution_highlight(score, title, color_index, colored_name='Coreset', x_range=None, y_range=None, filename=None):
    """
    Plot the distribution of data scores, with scores at color_index highlighted in red.

    Parameters:
    - score: Array of score data.
    - title: Title for the plot.
    - color_index: List or array of indices to highlight in the plot.
    - filename: Optional; if provided, the plot will be saved to this file.
    """
    
    _fontsize = 15
    title_fontsize = 17

    fig, ax = plt.subplots()
    ax.tick_params(axis="both", direction="in", length=2, width=1)

    # Plot all scores with low opacity
    n, x, _ = ax.hist(score, bins=30, alpha=0.1, label="All data", color='blue')
    n, x = np.histogram(score, 30)
    bin_centers = 0.5*(x[1:]+x[:-1])
    ax.plot(bin_centers, n, linewidth=1, c='#3a5bd4')  # Plotting the line for all scores

    # Highlight the scores at color_index
    coreset_scores = score[color_index]
    n_mislabel, x_mislabel, _ = ax.hist(coreset_scores, bins=x, alpha=0.5, label=colored_name, color='green', hatch='//')  

    x_min = int((score.min()//10 - 1) * 10)
    x_max = int((score.max()//10 + 1) * 10)
    if x_range:
        x_min, x_max = x_range[0], x_range[1]
    if y_range:
        ax.set_ylim(y_range[0], y_range[1])
    stride = calculate_stride(x_min, x_max)


    ax.tick_params(axis="both", direction="in", length=2, width=1)
    ax.set_xlabel('Area under the margin (AUM)', fontsize=title_fontsize)
    ax.set_ylabel('#Examples', fontsize=title_fontsize)
    ax.set_xlim(x_min, x_max)
    ax.grid(zorder=-10, color='#eaeaf2')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks(range(x_min, x_max, stride))
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    plt.rc('font', size=_fontsize)
    ax.legend(prop={'size': title_fontsize})
    ax.set_axisbelow(True)

    plt.title(title, fontsize=title_fontsize)

    if filename:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()


def plot_data_score_distribution(score, title, x_range=None, y_range=None, scale_200=False, filename=None):
    """
    Plot the distribution of data scores.

    Parameters:
    - path: Path to the pickle file containing score data.
    - title: Title for the plot.
    - filename: Optional; if provided, the plot will be saved to this file.
    """
    if scale_200:
        score = score / score.max() * 200
    _fontsize = 15
    title_fontsize = 17

    fig, ax = plt.subplots()
    ax.tick_params(axis="both", direction="in", length=2, width=1)


    n, x, _ = ax.hist(score, bins=30, alpha=0.1, label="All data")
    n, x = np.histogram(score, 30)
    bin_centers = 0.5*(x[1:]+x[:-1])
    ax.plot(bin_centers, n, linewidth=1, c='#3a5bd4')

    # closest 10 bin to x_min
    x_min = int((score.min()//10 - 1) * 10)
    x_max = int((score.max()//10 + 1) * 10)
    print(x_min, x_max)
    if x_range:
        x_min, x_max = x_range[0], x_range[1]
    if y_range:
        ax.set_ylim(y_range[0], y_range[1])
    stride = calculate_stride(x_min, x_max)

    ax.tick_params(axis="both", direction="in", length=2, width=1)
    ax.set_xlabel('Area under the margin (AUM)', fontsize=title_fontsize)
    ax.set_ylabel('#Examples', fontsize=title_fontsize)
    ax.set_xlim(x_min, x_max)
    ax.grid(zorder=-10, color='#eaeaf2')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks(range(x_min, x_max, stride))
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    plt.rc('font', size=_fontsize)
    ax.legend(prop={'size': title_fontsize})
    ax.set_axisbelow(True)

    plt.title(title, fontsize=title_fontsize)

    if filename:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()


def plot_pseudo_gt_aum(score, gt_score, mislabel_indices, title, point_size=10, draw_correct=True, draw_mis=True, draw_all=False):
    plt.figure(figsize=(10, 6))

    if draw_all:
        plt.scatter(score, gt_score, color='blue', s=point_size)

    else:
        # Plot correctly labeled points
        correct_indices = np.setdiff1d(np.arange(len(score)), mislabel_indices)
        if draw_correct:
            plt.scatter(score[correct_indices], gt_score[correct_indices], color='blue', label='Correctly labeled', s=point_size)
        
        # Plot mislabeled points
        if draw_mis:
            plt.scatter(score[mislabel_indices], gt_score[mislabel_indices], color='red', label='Mislabelled', s=point_size)
        
    # Plot a diagonal line to represent perfect alignment
    max_val = max(score.max(), gt_score.max())
    min_val = min(score.min(), gt_score.min())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black')
    
    plt.xlabel('Pseudo Label AUM')
    plt.ylabel('Ground Truth AUM')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

##############<-Clusterings and Embedings->##############

def visualize_distance_distribution(distances, misclassified_indices):
    """
    Visualizes the distribution of distances to class centroids, with an overlay for misclassified indices.
    
    Parameters:
    - distances: A tensor or numpy array containing the distances to the class centroids.
    - misclassified_indices: A list or array containing the indices of the misclassified samples.
    """
    # Convert distances to a numpy array if it's a tensor
    if not isinstance(distances, np.ndarray):
        distances = distances.numpy()
    
    # Extract distances for misclassified and correctly classified indices
    misclassified_distances = distances[misclassified_indices]
    correct_indices = np.setdiff1d(np.arange(len(distances)), misclassified_indices)
    correct_distances = distances[correct_indices]
    
    # Plot histogram of distances for all examples and misclassified examples
    plt.figure(figsize=(10, 6))
    # Histogram for correctly classified examples
    plt.hist(correct_distances, bins=30, alpha=0.5, label='Correctly Labeled', color='blue')
    # Histogram for misclassified examples
    plt.hist(misclassified_distances, bins=30, alpha=0.7, label='Mislabeled', color='red')
    
    plt.title('Distribution of Distances to Class Centroids (CIFAR-10)')
    plt.xlabel('Distance to Centroid')
    plt.ylabel('Number of Examples')
    plt.legend()
    plt.show()


def dist_to_centroid(embeds, pseudo_train_labels_tensor):
    embeds = torch.tensor(embeds, dtype=torch.float)
    pseudo_train_labels_tensor = torch.tensor(pseudo_train_labels_tensor, dtype=torch.long)
    
    num_classes = torch.unique(pseudo_train_labels_tensor).size(0)
    
    centroids = torch.zeros(num_classes, embeds.size(1))
    
    for class_idx in range(num_classes):
        class_mask = pseudo_train_labels_tensor == class_idx
        class_embeds = embeds[class_mask]
        centroids[class_idx] = class_embeds.mean(dim=0)
    
    distances = torch.zeros(embeds.size(0))
    for i, (embed, label) in enumerate(zip(embeds, pseudo_train_labels_tensor)):
        distances[i] = torch.norm(embed - centroids[label])
    
    return distances


def visualize_aum_distance_distribution(aum_score, distances, misclassified_indices):
    """
    Visualizes the relationship between AUM scores and distances to class centroids,
    highlighting misclassified examples.

    Parameters:
    - aum_score: A tensor or numpy array containing the AUM scores for each example.
    - distances: A tensor or numpy array containing the distances to the class centroids for each example.
    - misclassified_indices: A list or array containing the indices of the misclassified samples.
    """
    if not isinstance(aum_score, np.ndarray):
        aum_score = aum_score.numpy()
    if not isinstance(distances, np.ndarray):
        distances = distances.numpy()
    
    all_indices = np.arange(len(distances))
    correctly_classified_indices = np.setdiff1d(all_indices, misclassified_indices)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for correctly classified examples
    plt.scatter(distances[correctly_classified_indices], aum_score[correctly_classified_indices], 
                color='blue', alpha=0.2, label='Correctly Classified')
    
    # Scatter plot for misclassified examples
    plt.scatter(distances[misclassified_indices], aum_score[misclassified_indices], 
                color='red', alpha=0.2, label='Misclassified')
    
    # Adding labels and title
    plt.xlabel('Distance to Centroid')
    plt.ylabel('AUM Score')
    plt.title('AUM Score vs. Distance to Class Centroid')
    plt.legend(loc='upper right')
    
    plt.show()
    plt.grid(True)




##############<-CIFAR-10/CIFAR-100 Visualization->##############
        
def describ_dataset(trainset, num_images_per_class, max_class_num=10):
    num_images_per_class = 3

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)


    # Select three images for each class
    selected_images = {cls: [] for cls in range(max_class_num)}
    for images, labels in trainloader:
        label = labels.item()
        if len(selected_images[label]) < num_images_per_class:
            selected_images[label].append(images[0].numpy())
        if all(len(imgs) == num_images_per_class for imgs in selected_images.values()):
            break
    plot_images_by_class(selected_images, range(0, 10))
    

def describe_selected_images(trainset, num_images_per_class, index):
    """
    Select and describe specific images from the dataset based on the provided indices.

    Parameters:
    - trainset: The dataset from which to select images.
    - num_images_per_class: The number of images to select per class.
    - index: A list of indices specifying which samples to consider for selection.
    """

    
    # Initialize the DataLoader without shuffling, since we're selecting specific samples
    cls = set([trainset.targets[i] for i in index])
    trainset = Subset(trainset, index)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

    
    selected_images = {c: [] for c in cls}

    for i, (images, labels) in enumerate(trainloader):
        label = labels.item()
        # Check if the current class needs more images
        if len(selected_images[label]) < num_images_per_class:
            selected_images[label].append(images[0].numpy())
        # Break if each class has enough images
        if all(len(imgs) == num_images_per_class for imgs in selected_images.values()):
            break

    # Assuming plot_images_by_class is a function you have defined to plot images
    plot_images_by_class(selected_images, range(0, 10))




def plot_images_by_class(selected_images, class_names):
    for cls, images in selected_images.items():
        # Skip classes with no images
        if not images:
            print(f"Class {class_names[cls]} has no images to display.")
            continue
        
        cls_name = class_names[cls]
        print(f"Class: {cls_name}")
        num_images = len(images)
        # Adjust subplot for classes with fewer images
        cols = max(1, num_images)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 2, 2), squeeze=False)
        
        for i, ax in enumerate(axes.flatten()):
            if i < num_images:
                img = images[i]
                if img.ndim == 3 and img.shape[0] in {1, 3}:
                    img = img.transpose(1, 2, 0)  
                if img.min() < 0 or img.max() > 1:
                    img = np.clip(img / 2 + 0.5, 0, 1) 
                ax.imshow(img)
            ax.axis('off')
        
        plt.show()