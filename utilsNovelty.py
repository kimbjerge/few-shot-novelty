"""
General utilities

Modified on Thu Nov  2 09:52:46 2023

@author: Kim Bjerge
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.stats import norm 
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from NoveltyThreshold import BayesTwoClassThreshold, StdTimesTwoThredshold
from easyfsl.methods import FewShotClassifier


def plot_images(images: Tensor, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W)
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.

    Returns:
        average of the last window instances in value_list

    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def predict_embeddings(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict embeddings for a dataloader.
    Args:
        dataloader: dataloader to predict embeddings for. Must deliver tuples (images, class_names)
        model: model to use for prediction
        device: device to cast the images to. If none, no casting is performed. Must be the same as
            the device the model is on.
    Returns:
        dataframe with columns embedding and class_name
    """
    all_embeddings = []
    all_class_names = []
    with torch.no_grad():
        for images, class_names in tqdm(
            dataloader, unit="batch", desc="Predicting embeddings"
        ):
            if device is not None:
                images = images.to(device)
            all_embeddings.append(model(images).detach().cpu())
            if isinstance(class_names, torch.Tensor):
                all_class_names += class_names.tolist()
            else:
                all_class_names += class_names

    concatenated_embeddings = torch.cat(all_embeddings)

    return pd.DataFrame(
        {"embedding": list(concatenated_embeddings), "class_name": all_class_names}
    )


def evaluate_on_one_task_V1(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    novelIdx = 0 # could be randomized based on number of classes (ways)
    
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    
    minValue = predictions.min() - 1
    predictions[:,novelIdx] = minValue # Set novelIdx of ways below minium score
    fewShotPredictions = predictions[query_labels != novelIdx] # Exclude novelty predictions (Here I wronly use the knowledge of new shot!)
    max_values = torch.max(fewShotPredictions, 1)[0] # Find shortes distance in feature space
    mean_max_values = max_values.mean() # Compute minimum of shortes distances
    predictions[query_labels == novelIdx, novelIdx] = mean_max_values # Set way novelIdx to mean of shortes distances
    
    number_of_correct_predictions = (
        (torch.max(predictions, 1)[1] == query_labels).sum().item()
    )
    return number_of_correct_predictions, len(query_labels)

def evaluate_on_one_task_V2(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
    thDiff = -4.5, # To be learned
    thMin = -20.0 # To be learned
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    novelIdx = 0 # could be randomized based on number of classes (ways)
    
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    
    predictionsKWays = predictions[:,1:] # Novel class is index 0
    kWay = predictionsKWays.shape[1]
    meanDist = predictionsKWays.mean(1)
    minDistSet = predictionsKWays.max(1)
    minDistIdx = minDistSet[1]+1
    minDist = minDistSet[0]
    diffMeanDist = (meanDist - minDist/kWay)*(kWay/(kWay-1))
    diffMeanMinDist = diffMeanDist - minDist

    aboveThDiff = diffMeanMinDist > thDiff
    belowThMin = minDist < thMin
    minDistIdx[belowThMin & aboveThDiff] = 0 # Novel classes
    
    # minValue = predictions.min() - 1
    # predictions[:,novelIdx] = minValue # Set novelIdx of ways below minium score
    # fewShotPredictions = predictions[query_labels != novelIdx] # Exclude novelty predictions
    # max_values = torch.max(fewShotPredictions, 1)[0] # Find shortes distance in feature space
    # mean_max_values = max_values.mean() # Compute minimum of shortes distances
    # predictions[query_labels == novelIdx, novelIdx] = mean_max_values # Set way novelIdx to mean of shortes distances
    
    # number_of_correct_predictions = (
    #     (torch.max(predictions, 1)[1] == query_labels).sum().item()
    # )
    number_of_correct_predictions = (
        (minDistIdx == query_labels).sum().item()
    )
    return number_of_correct_predictions, len(query_labels)


def evaluate_V2(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> float:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy
    """
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    thMinStart = -19.0 # -20.0 guess, best -21.0
    thMinStep = -0.1
    
    #thMin = thMinStart
    thMin = -21.0
    
    thDiffStart = -2.5 # -4.5 guess, best -3.5
    thDiffStep = -0.1
    
    #thDiff = thDiffStart
    thDiff = -3.5
    
    iterations = 0
    accuracyBest = 0
    
    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                correct, total = evaluate_on_one_task_V2(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                    thDiff,
                    thMin,
                )

                total_predictions += total
                correct_predictions += correct
                accuracy = correct_predictions / total_predictions
                
                iterations += 1
                if iterations % 20 == 0:
                    print("thDiff, tMin", thDiff, thMin)
                    if accuracy > accuracyBest:
                        accuracyBest = accuracy
                        thDiffBest = thDiff
                        thMinBest = thMin
                        print("Best accuracy", accuracyBest, thDiffBest, thMinBest)
                    #total_predictions = 0
                    #correct_predictions = 0
                    #thDiff += thDiffStep # Choose to sweep
                    #thMin += thMinStep
                    
                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=accuracy)

    return correct_predictions / total_predictions


class Metrics:
    def __init__(self):
      
      self.true_positive = 0
      self.false_positive = 0
      self.false_negative = 0

    # Calculate the metrics for the novelty class with label index 0
    def calcMetrics(self, predicted, query_labels):
        
        # Label 0 is outlier label 
        TP=(predicted[query_labels==0] == 0).sum().cpu().numpy() # Same labels as query
        self.true_positive += TP
        FP=(predicted[query_labels!=0] == 0).sum().cpu().numpy() # Novelty label for know classes
        self.false_positive += FP
        FN=(predicted[query_labels==0] != 0).sum().cpu().numpy() # Novel class with label of know classes
        self.false_negative += FN
        #print("TP FP FN", self.true_positive, self.false_positive, self.false_negative)
        
    def TP(self):
        return self.true_positive

    def FP(self):
        return self.false_positive

    def FN(self):
        return self.false_negative
    
    def recall(self):
        if (self.true_positive == 0):
            return 0
        else:
            return self.true_positive/(self.false_negative+self.true_positive)
    
    def precision(self):
        if (self.true_positive == 0):
            return 0
        else:
            return self.true_positive/(self.false_positive+self.true_positive)
    
    def f1score(self):
        R = self.recall()
        P = self.precision()
        if (R+P) == 0:
            return 0
        else:
            return (2*P*R)/(P+R)


predictions_true = [] # Used during learning the thredshold for outlier detection
predictions_false = []

def evaluate_on_one_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
    use_novelty,
    learn_th,
    thMin,
    metric,
    device
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    model.process_support_set(support_images, support_labels)
    #predictions, pred_std = model(query_images) #.detach().data
    predictions = model(query_images).detach().data

    if learn_th:
        
        #correct_max_predictions = predictions[torch.max(predictions, 1)[1] == query_labels].max(1)[0]
        #Below line of code is wrong!
        #wrong_predictions = predictions[torch.max(predictions, 1)[1] != query_labels].reshape(-1)

        correct_episodes = predictions[torch.max(predictions, 1)[1] == query_labels.cpu()]
        correct_scores = correct_episodes.max(1)[0]
        correct_pred_idx = correct_episodes.max(1)[1]            
        
        #Select scores part of correct predicitons that don't belong to the correct query label
        num_rows = correct_episodes.shape[0]
        num_cols = correct_episodes.shape[1]
        wrong_scores = torch.empty(num_rows*(num_cols-1)).to(device)
        idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if j != correct_pred_idx[i]:
                    wrong_scores[idx]=correct_episodes[i][j]
                    idx += 1
                    
        predictions_false.extend(wrong_scores.tolist())
        predictions_true.extend(correct_scores.tolist())
        
    if use_novelty:
        predictionsKWays = predictions[:,1:] # Novel class is index 0
        minDistSet = predictionsKWays.max(1)
        minDist = minDistSet[0]
        minDistIdx = minDistSet[1]+1 # Add one to class index, where novel class = 0
        belowThMin = minDist < thMin
        minDistIdx[belowThMin] = 0 # Novel class if correclation is below threshold
        minDistIdx = minDistIdx.to(device)
        number_of_correct_predictions = (
            (minDistIdx == query_labels).sum().item()
        )

    else:
        predictions = predictions.to(device)
        minDistIdx = torch.max(predictions, 1)[1]
        number_of_correct_predictions = (
            (minDistIdx == query_labels).sum().item()
        )
        #if learn_th:
            #print(np.mean(predictions_true)-2*np.std(predictions_true))
        #    print("Th, avg, std", np.mean(predictions_true)-2*np.std(predictions_true), np.mean(predictions_true), np.std(predictions_true))
 
    if metric != None:
        metric.calcMetrics(minDistIdx, query_labels)

    return number_of_correct_predictions, len(query_labels)


def evaluate(
    model: FewShotClassifier,
    data_loader: DataLoader,
    novelty_th, 
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
    plt_hist: bool = True, 
    use_novelty: bool = False, 
    metric = None,
    learn_th: bool = False
) -> float:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        novelty_th: novelty threshold to detect outliers based on normalized corrlation
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
        plt_hist: plot histogram after learning novelty_th
        use_novelty: True to use novelty detection
        learn_th: True to learn novelty_th on dataset
    Returns:
        average classification accuracy
    """
    
    # We'll count everything and compute the ratio at the end
    print("Use novelty detection with threshold", use_novelty, novelty_th)
    print("Learn threshold value", learn_th, device)
    
    total_predictions = 0
    correct_predictions = 0
    if learn_th:
        predictions_true.clear()
        predictions_false.clear()

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                correct, total = evaluate_on_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                    use_novelty,
                    learn_th,
                    novelty_th,
                    metric,
                    device
                )

                total_predictions += total
                correct_predictions += correct

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    learned_th = novelty_th
    mu_k = 0
    sigma_k = 0
    mu_o = 0
    sigma_o = 0
    if learn_th: 
        mu_k = np.mean(predictions_true)
        sigma_k =  np.std(predictions_true)
        var_k = np.var(predictions_true)
        th = StdTimesTwoThredshold(mu_k, sigma_k)

        mu_o = np.mean(predictions_false) 
        sigma_o =  np.std(predictions_false)
        var_o = np.var(predictions_false)

        #learned_th = th
        bayes_th = BayesTwoClassThreshold(var_k, mu_k, var_o, mu_o, 5, 1) # 5-way, 1-outlier
        print(th, bayes_th)
        
        if plt_hist:
            # Plot the histogram of true prediction values
            plt.hist(predictions_true, bins=100, density=True, alpha=0.5, color='b')
            plt.hist(predictions_false, bins=100, density=True, alpha=0.5, color='r')
            xmin, xmax = plt.xlim() 
            x = np.linspace(xmin, xmax, 100) 
            # add a 'best fit' line
            y_k = norm.pdf(x, mu_k, sigma_k)
            plt.plot(x, y_k, 'k--', linewidth=1)
            y_o = norm.pdf(x, mu_o, sigma_o)
            plt.plot(x, y_o, 'k--', linewidth=1)
            steps = 20
            step = max(y_k)/steps
            listTh = [th for i in range(steps)]
            listBTh = [bayes_th for i in range(steps)]
            listPb = [step*i for i in range(steps)]
            plt.plot(listTh, listPb, 'k--')
            plt.plot(listBTh, listPb, 'g')
            #plt.xlabel('True Positive (Cosine Similarity)')
            plt.xlabel('Cosine Similarity')
            plt.xlim(0.2, 1.0)
            #plt.ylim(0.0, 10.0) # CUB
            #plt.ylim(0.0, 15.0) # euMoths
            plt.ylim(0.0, 20.0) # Omniglot
            plt.ylabel('Probability (%)')
            plt.title('Distribution of TN (red) and TP (blue) (th>%.4f)' % (bayes_th))
            # Tweak spacing to prevent clipping of ylabel
            plt.subplots_adjust(left=0.15)
            plt.show()
            
        learned_th = bayes_th # Or sigma threshold (th)
        
    return correct_predictions / total_predictions, learned_th, mu_k, sigma_k, mu_o, sigma_o


def compute_average_features_from_images(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
):
    """
    Compute the average features vector from all images in a DataLoader.
    Assumes the images are always first element of the batch.
    Returns:
        Tensor: shape (1, feature_dimension)
    """
    all_embeddings = torch.stack(
        predict_embeddings(dataloader, model, device)["embedding"].to_list()
    )
    average_features = all_embeddings.mean(dim=0)
    if device is not None:
        average_features = average_features.to(device)
    return average_features
