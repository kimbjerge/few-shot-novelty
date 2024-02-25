"""
General utilities

Modified on Sun Feb 25 11:34:31 2024 from easy-few-shot-learning

@author: Kim Bjerge
"""
from typing import Optional, Tuple
import numpy as np
import torch
from scipy.stats import norm 
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from NoveltyThreshold import BayesTwoClassThreshold, StdTimesTwoThredshold
from easyfsl.methods import FewShotClassifier

usePriorProbabilities = True # Set to False for testing p(o) = p(k) (Outliers = Known classes)
#usePriorProbabilities = False # Set to False for testing p(o) = p(k) (Outliers = Known classes)

# Class used during learning to compute precision, recall and F1-score for outlier class (labels == 0)
class Metrics:
    def __init__(self):
      
      self.true_positive = 0
      self.false_positive = 0
      self.false_negative = 0

    # Calculate the metrics for the novelty class with label index 0
    def calcMetrics(self, predicted, query_labels, novelClassId=0):
        
        # Label 0 is outlier label 
        TP=(predicted[query_labels==novelClassId] == 0).sum().cpu().numpy() # Same labels as query
        self.true_positive += TP
        FP=(predicted[query_labels!=novelClassId] == 0).sum().cpu().numpy() # Novelty label for know classes
        self.false_positive += FP
        FN=(predicted[query_labels==novelClassId] != 0).sum().cpu().numpy() # Novel class with label of know classes
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

# NB! Global variables used to learn distributions of true and false predicitons during learning phase
predictions_true = [] # Used during learning the thredshold for outlier detection
predictions_false = []

def evaluate_on_one_task_M_novel(
    model: FewShotClassifier,
    support_images: Tensor, # Contains n_way + n_novel classes, but only n_way support classes are used
    support_labels: Tensor,
    query_images: Tensor, # Contains n_way + n_novel classes
    query_labels: Tensor,
    use_novelty,
    learn_th,
    thMin,
    n_way,
    metric,
    device,
    n_novel = 1
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    assert(n_way > 0) # n_way
    
    # n_shot = len(support_labels) / (n_way+n_novel)
    # n_query = len(query_labels) / (n_way+n_novel)

    novelClassId = 0 # ClassId for novel classes
    if use_novelty and n_novel > 0:       
        # Remove novel classes from support set
        # Classes removed from support set starting with lowest ClassIds
        support_to_use = support_labels != novelClassId # Start removing first classId = 0
        for nClassId in range(n_novel-1): # More than one novel class
            to_use = support_labels != nClassId+1 # novelClassId == [0, 1, 2, .. n_novel-1]
            support_to_use = support_to_use.logical_and(to_use)          
        support_images = support_images[support_to_use,:]
        # Renumber support labels ClassId == [1, 2, 3, 4 ..]
        support_labels = support_labels[support_to_use] - n_novel 
        
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data

    if learn_th:
        # Learning threshold based FSL similarity distributions
        correct_episodes = predictions[torch.max(predictions, 1)[1] == query_labels]
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
        
    if use_novelty and n_novel > 0:
        # Few-shot-novelty learning (FSNL)
        minDistSet = predictions.max(1)
        minDist = minDistSet[0]
        minDistIdx = minDistSet[1] + 1 
        # Novel class if below similarty threshold
        belowThMin = minDist < thMin 
        minDistIdx[belowThMin] = novelClassId # Set novelClassId = 0 if below threshold
        minDistIdx = minDistIdx.to(device)
        # Renumber query labels ClassId == [1, 2, 3, .. ]
        for idx in range(len(query_labels)):
            if query_labels[idx] >= n_novel:
                query_labels[idx] = query_labels[idx] - (n_novel-1) # Renumber label
            else:
                query_labels[idx] = novelClassId # All novel classes labeled with novelClassId = 0
                
        number_of_correct_predictions = (
            (minDistIdx == query_labels).sum().item()
        )

    else:
        # Few-shot learning (FSL)
        predictions = predictions.to(device)
        minDistIdx = torch.max(predictions, 1)[1]
        number_of_correct_predictions = (
            (minDistIdx == query_labels).sum().item()
        )
 
    if metric != None:
        metric.calcMetrics(minDistIdx, query_labels, novelClassId=novelClassId)

    return number_of_correct_predictions, len(query_labels)


def evaluate_M_novel(
    model: FewShotClassifier,
    data_loader: DataLoader,
    novelty_th, 
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
    plt_hist: bool = True, 
    use_novelty: bool = False,
    n_way = 5,
    metric = None,
    learn_th: bool = False,
    n_novel = 1
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
        n_way: The number of classes in support set (default 5 way)
        metric: Metric class to compute precision and recall for novelty class
        learn_th: True to learn novelty_th on dataset
    Returns:
        average classification accuracy, learned_th, mu_k, sigma_k, mu_o, sigma_o
    """
        
    total_predictions = 0
    correct_predictions = 0
    if learn_th:
        print("Learn threshold value with k-way", n_way, "and m-novel", n_novel)
        predictions_true.clear()
        predictions_false.clear()
    else:
        print("Use novelty detection with k-way", n_way, "and m-novel", n_novel, use_novelty, "TH", novelty_th)

    print("On device", device)
 
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
                correct, total = evaluate_on_one_task_M_novel(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                    use_novelty,
                    learn_th,
                    novelty_th,
                    n_way,
                    metric,
                    device,
                    n_novel
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
        if usePriorProbabilities:
            bayes_th, x2 = BayesTwoClassThreshold(var_k, mu_k, var_o, mu_o, n_way, n_novel) # 5-way, 1-novel classes (default)
        else:
            bayes_th, x2 = BayesTwoClassThreshold(var_k, mu_k, var_o, mu_o, 1, 1) # p(o) = p(k)
            
        print("Threshold", th, bayes_th, "Prior", usePriorProbabilities)
        
        if plt_hist:
            # Plot the histogram of true prediction values
            plt.hist(predictions_true, bins=100, density=True, alpha=0.5, color='b')
            plt.hist(predictions_false, bins=100, density=True, alpha=0.5, color='r')
            xmin, xmax = plt.xlim() 
            x = np.linspace(xmin, xmax, 100) 
            # add a 'best fit' line
            y_k = norm.pdf(x, mu_k, sigma_k)
            plt.plot(x, y_k, 'b--', linewidth=1)
            y_o = norm.pdf(x, mu_o, sigma_o)
            plt.plot(x, y_o, 'r--', linewidth=1)
            steps = 20
            step = max(y_k)/steps
            listTh = [th for i in range(steps)]
            listBTh = [bayes_th for i in range(steps)]
            listX2 = [x2 for i in range(steps)]
            listPb = [step*i for i in range(steps)]
            #plt.plot(listTh, listPb, 'k--')
            plt.plot(listBTh, listPb, 'k')
            #plt.plot(listX2, listPb, 'k--')
            #plt.xlabel('True Positive (Cosine Similarity)')
            plt.xlabel('Cosine similarity')
            #plt.xlim(0.2, 1.0)
            #plt.ylim(0.0, 10.0) # CUB
            #plt.ylim(0.0, 12.0) # miniImagenet
            #plt.ylim(0.0, 15.0) # euMoths
            #plt.ylim(0.0, 20.0) # Omniglot
            plt.ylabel('Probability (%)')
            #plt.title('Pre-trained on ImageNet')
            #plt.title('Fine-tuned on EU Moths')
            #plt.title('Distribution of similarities (th=%.4f)' % (bayes_th))
            # Tweak spacing to prevent clipping of ylabel
            plt.legend(["True positive", "True negative"])
            plt.subplots_adjust(left=0.15)
            plt.show()
            
        learned_th = bayes_th # Or sigma threshold (th)
        
    return correct_predictions / total_predictions, learned_th, mu_k, sigma_k, mu_o, sigma_o
