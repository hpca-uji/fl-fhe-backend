import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, roc_curve, auc
from .error import ErrorHandler
from typing import OrderedDict, List
from src.nn.layers.encrypted_layer import encrypt_weights
from src.nn.layers.layer_util import deserialized_model, deserialized_layer


def choice_device(device):
    """
    A function to choose the device

    :param device: the device to choose (cpu, gpu or mps)
    """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and device != "cpu":
        """
        on Mac : 
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    return device


def check_directory(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

def load_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        raise ErrorHandler("File not found", 0)
    
def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            query_str = pickle.load(file)
        contexte = query_str["contexte"]
        del query_str["contexte"]
        return query_str, contexte

    else:
        raise ErrorHandler("File not found", 0)

def write_file(file_path, client_query):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
    with open(file_path, 'wb') as file:  # 'ab' to add existing file
        encode_str = pickle.dumps(client_query)
        file.write(encode_str)

def save_matrix(y_true, y_pred, path, classes):
    check_directory(path)
    y_true_mapped = [classes[label] for label in y_true]
    y_pred_mapped = [classes[label] for label in y_pred]
    
    cf_matrix_normalized = confusion_matrix(y_true_mapped, y_pred_mapped, labels=classes, normalize='all')
   
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    df_cm = pd.DataFrame(cf_matrix_round, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    plt.savefig(path)
    plt.close()

def save_roc(targets, y_proba, path, num_classes):
    check_directory(path)
    y_true = np.zeros(shape=(len(targets), num_classes))  # array-like of shape (n_samples, n_classes)
    for i in range(len(targets)):
        y_true[i, targets[i]] = 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    lw = 2
    for i in range(num_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label='Worst case')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) Curve OvR")  # One vs Rest
    plt.legend(loc="lower right")  # loc="best"

    plt.savefig(path)
    plt.close()

def save_graphs(plot_path, local_epoch, results, end_file=""):
    check_directory(plot_path)
    os.makedirs(plot_path, exist_ok=True)  # to create folders results
    print("save graph in ", plot_path)
    # plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc"], results["val_acc"]],
        "Epochs", "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=plot_path + "Accuracy_curves" + end_file)

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss"], results["val_loss"]],
        "Epochs", "Loss",
        curve_labels=["Training loss", "Validation loss"], title="Loss curves",
        path=plot_path + "Loss_curves" + end_file)
    
def plot_graph(list_xplot, list_yplot, x_label, y_label, curve_labels, title, path=None):
    check_directory(path)
    lw = 2

    plt.figure()
    for i in range(len(curve_labels)):
        plt.plot(list_xplot[i], list_yplot[i], lw=lw, label=curve_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if curve_labels:
        plt.legend(loc="lower right")

    if path:
        plt.savefig(path)

def get_parameters2(net, context_client=None) -> List[np.ndarray]:
    """
    Get the parameters of the network
    :param net: network to get the parameters (weights and biases)
    :param context_client: context of the crypted weights (if None, return the clear weights)
    :return: list of parameters (weights and biases) of the network
    """
    if context_client:
        # Crypte of the model trained at the client for a given round (after each round the model is aggregated between
        # clients)
        encrypted_tensor = encrypt_weights(net.state_dict(), context_client)  # list of encrypted layers (weights and biases)

        return [layer.get_weight() for layer in encrypted_tensor]

    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray], context_client=None):
    """
    Update the parameters of the network with the given parameters (weights and biases)
    :param net: network to set the parameters (weights and biases)
    :param parameters: list of parameters (weights and biases) to set
    :param context_client: context of the crypted weights (if None, set the clear weights)
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    if context_client:
        secret_key = context_client.secret_key()
        dico = {k: deserialized_layer(k, v, context_client) for k, v in params_dict}

        state_dict = OrderedDict(
            {k: torch.Tensor(v.decrypt(secret_key)) for k, v in dico.items()}
        )

    else:
        dico = {k: torch.Tensor(v) for k, v in params_dict}
        state_dict = OrderedDict(dico)

    net.load_state_dict(state_dict, strict=True)
    print("Updated model")