import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve)

FIGS_DIR = os.path.join(os.getcwd(), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

def predictions(model, dataloader, device) : 
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in dataloader : 
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = torch.softmax(logits, dim = 1)
            preds = probs.argmax(dim = 1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        return (np.concatenate(all_preds), np.concatenate(all_labels), np.vstack(all_probs))
    

def print_report(y_true, y_pred, target_names = None) :
    report = classification_report(y_true, y_pred, target_names = target_names, digits = 4)
    print("\nClassification report\n", report)



def create_confusion_matrix(y_true, y_pred, classes, save_name = "confusion_matrix.png") :
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis = 1)[: , np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation='nearest', cmap = plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks = np.arange(len(classes)),
        yticks = np.arange(len(classes)), 
        xticklabels = classes,
        yticklabels = classes,
        ylabel = 'True label',
        xlabel = 'Predicted label',
        title = 'Confusion Matrix'
    )



    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]) :
        for j in range(cm_norm.shape[1]) :
            ax.text(
                j, i,
                f"{cm[i,j]}",
                ha="center", va="center",
                color = "white" if cm_norm[i,j] > thresh else "black"
            )


    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, save_name), bbox_inches="tight")
    plt.show()




def plot_roc(y_true, y_probs, classes) : 

    y_true_oh =  np.eye(len(classes))[y_true]

    plt.figure()
    for i, cls in enumerate(classes) :
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_probs[:, i])
        auc = roc_auc_score(y_true_oh[:,i], y_probs[:,i])
        plt.plot(fpr, tpr, label = f"{cls} (AUC = {auc:.3f}) ")
        plt.plot([0,1], [0,1], 'k--', label="Chance")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curves")
        plt.legend(loc = "best")
        save_name = f"roc_plot_{cls}.png"
        plt.savefig(os.path.join(FIGS_DIR, save_name), bbox_inches="tight")
        plt.show()

def plot_prob_correlation_matrix(y_probs, classes, save_name = "correlation_matrix.png"):
    df = pd.DataFrame(y_probs, columns=classes)
    corr = df.corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.colorbar(cax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title="Probability Correlation Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, save_name), bbox_inches="tight")
    plt.show()


def evaluate_model(model, dataloader, device, class_names):
    y_pred, y_true, y_probs = predictions(model, dataloader, device)
    print_report(y_true, y_pred, target_names=class_names)
    create_confusion_matrix(y_true, y_pred, class_names)
    plot_roc(y_true, y_probs, class_names)
    plot_prob_correlation_matrix(y_probs, class_names)
