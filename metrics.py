import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score

def plot_TSNE(num_classes, features, labels, categories, path=None):
    """
    Plots a t-SNE plot for the given features and labels.
    """
    X = np.array(features)

    tsne = TSNE(metric='cosine').fit_transform(X)

    df = pd.DataFrame({
        'x': tsne[:, 0],
        'y': tsne[:, 1],
        'labels': labels,
        'categories': categories
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='labels',
        style='categories',
        palette=sns.color_palette("husl", num_classes),
        alpha=0.7,
        legend='full'
    )

    plt.title("TSNE")
    #plt.legend(title='Cluster Label')
    plt.legend([],[], frameon=False)

    if path is not None:
        plt.savefig(path)

    plt.show()

def plot_confusion(num_classes, predictions, targets, title, path=None):
    """
    Plots a confusion matrix for the given predictions and targets.
    """

    confmat = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true')
    confusion = confmat(torch.tensor(predictions), torch.tensor(targets))

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion.numpy(), annot=True, fmt=".2f", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes), annot_kws={"size": 8})

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)

    plt.show()

def compute_accuracy(num_classes, predictions, targets):
    """
    Computes the top-1-accuracy for the given predictions and targets.
    """

    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
    accuracy = accuracy_metric(torch.tensor(predictions), torch.tensor(targets))

    return accuracy

def compute_f1_score(num_classes, predictions, targets):
    """
    Computes the F1-score for the given predictions and targets.
    """

    f1_metric = MulticlassF1Score(num_classes=num_classes, average='micro')
    f1_score = f1_metric(torch.tensor(predictions), torch.tensor(targets))

    return f1_score

def _compute_average_precision(relevance):
    """
    Computes the average precision for the given relevance.
    """

    num_relevant = relevance.sum()
    if num_relevant == 0:
        return 0

    precisions = []
    for k in range(1, len(relevance) + 1):
        if relevance[k - 1] == 1:
            prec = relevance[:k].sum() / k
            precisions.append(prec)

    return np.sum(precisions) / num_relevant

def compute_mean_average_precision(relevance_vector):
    """
    Computes the mean average precision for the given relevance vector.
    """

    return np.mean([_compute_average_precision(relevance) for relevance in relevance_vector])