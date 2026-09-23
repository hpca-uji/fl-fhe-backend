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
        context = query_str["context"]
        del query_str["context"]
        return query_str, context

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
    
    # Ensure inputs are numpy arrays and FLATTENED
    if hasattr(y_true, "numpy"): # Handle Torch Tensors
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "numpy"):
        y_pred = y_pred.detach().cpu().numpy()

    # For Multi-label, collapse to a binary 2x2 matrix (True/False Positives/Negatives across all labels)
    y_true_flat = np.asarray(y_true).ravel()
    y_pred_flat = (np.asarray(y_pred) > 0.5).astype(int).ravel() # Ensure predictions are binary
    
    cf_matrix_normalized = confusion_matrix(y_true_flat, y_pred_flat, normalize='all')
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    df_cm = pd.DataFrame(cf_matrix_round, index=["Negative", "Positive"], columns=["Negative", "Positive"])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    plt.savefig(path)
    plt.close()

def save_roc(targets, y_proba, path, num_classes):
    check_directory(path)
    
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    else:
        targets = np.asarray(targets)
    y_true = targets.astype(int) # targets is already shape (Batch, num_classes)
    
    if torch.is_tensor(y_proba):
        y_proba = y_proba.detach().cpu().numpy()

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
    :param context_client: FHE backend instance (if None, return the clear weights)
    :return: list of parameters (weights and biases) of the network
    """
    if context_client:
        he_backend = context_client
        encrypted = []
        for _, val in net.state_dict().items():
            enc_val = he_backend.encrypt(val.cpu().numpy())
            if hasattr(he_backend, "serialize"):
                enc_val = he_backend.serialize(enc_val)
            encrypted.append(enc_val)
            
            # Aggressively collect garbage to free intermediate FHE objects layer-by-layer
            import gc; gc.collect()
        return encrypted

    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray], context_client=None):
    """
    Update the parameters of the network with the given parameters (weights and biases)
    :param net: network to set the parameters (weights and biases)
    :param parameters: list of parameters (weights and biases) to set
    :param context_client: context of the crypted weights (if None, set the clear weights)
    """
    if context_client:
        he_backend = context_client
        state_dict = OrderedDict()
        keys = list(net.state_dict().keys())
        for i in range(len(keys)):
            k = keys[i]
            v = parameters[i]
            # Release original reference to free memory early
            parameters[i] = None
            
            # Flower may restore a single byte object as a 0-D numpy array
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
                
            # If the parameter is an unencrypted numpy array of numbers, skip decryption
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                dec = v
            elif isinstance(v, (int, float, np.number)):
                dec = v
            else:
                if hasattr(he_backend, "deserialize"):
                    v = he_backend.deserialize(v)
                dec = he_backend.decrypt(v)
            
            # Get target shape from the network state dict
            target_shape = net.state_dict()[k].shape
            num_elements = int(np.prod(target_shape))
            
            # Reshape the flattened decrypted array, truncating any padding elements
            dec_reshaped = np.array(dec).flatten()[:num_elements].reshape(target_shape)
            state_dict[k] = torch.tensor(dec_reshaped, dtype=net.state_dict()[k].dtype)
            
            # Force GC per layer to keep memory footprint extremely flat
            import gc; gc.collect()
    else:
        state_dict = OrderedDict()
        for k, v in zip(net.state_dict().keys(), parameters):
            state_dict[k] = torch.tensor(v, dtype=net.state_dict()[k].dtype)

    net.load_state_dict(state_dict, strict=True)
    print("Updated model")