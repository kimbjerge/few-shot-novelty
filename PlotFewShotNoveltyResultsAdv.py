# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:45:58 2023

@author: Kim Bjerge
"""

import matplotlib.pyplot as plt
import pandas as pd

resultDir = "./modelsFinalPreAdv/test_5w/" # 500*6 queries made on GPU cluser, 

def plotAccuracy(dataset, model, n_shot):
    title = dataset + " " + model + " (n-shot=" + str(n_shot) + ")"
    
    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    #data_few_shot = resultDir + model + "_" + dataset + "_novelty_learn.txt"
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName
    data_few_shot = resultDir + model + "_" + dataset + "_novelty_test.txt"
    
    data_few_shot_df = pd.read_csv(data_few_shot)
    data_novelty_df = data_few_shot_df[data_few_shot_df['Novelty'] == True]
    data_few_shot_df = data_few_shot_df[data_few_shot_df['Novelty'] == False]
    data_novelty_df = data_novelty_df[data_novelty_df['Model'] == model]
    
    data_df = data_novelty_df.loc[data_novelty_df['Shot'] == n_shot]
    data_few_shot_df = data_few_shot_df.loc[data_few_shot_df['Shot'] == n_shot]
    data_bayes_df = data_df.loc[data_df['Method'] == 'bayes']
    #data_std_df = data_df.loc[data_df['Method'] == 'std']
 
    ax = plt.gca()
    # Plot bayes vs. std
    data_few_shot_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='green', ax=ax)
    #data_std_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='blue', ax=ax)
    data_bayes_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='red', title=title, ax=ax)
    ax.set_ylabel("Accuracy")
    ax.legend(["FewShot", "Novelty"])
    plt.show()
    

def plotPrecisionRecall(dataset, model, n_shot):
    title = dataset + " " + model + " (n-shot=" + str(n_shot) + ")"
    
    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    #data_few_shot = resultDir + model + "_" + dataset + "_novelty_learn.txt"
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName
    data_few_shot = resultDir + model + "_" + dataset + "_novelty_test.txt"
    
    data_few_shot_df = pd.read_csv(data_few_shot)
    data_novelty_df = data_few_shot_df[data_few_shot_df['Novelty'] == True]
    data_few_shot_df = data_few_shot_df[data_few_shot_df['Novelty'] == False]
    data_novelty_df = data_novelty_df[data_novelty_df['Model'] == model]
    
    data_df = data_novelty_df.loc[data_novelty_df['Shot'] == n_shot]
    data_few_shot_df = data_few_shot_df.loc[data_few_shot_df['Shot'] == n_shot]
    data_bayes_df = data_df.loc[data_df['Method'] == 'bayes']
 
    ax = plt.gca()
    # Plot bayes vs. std
    data_few_shot_df.plot(kind='scatter', x='FewShotClassifier',  y='Precision',  color='green', ax=ax)
    data_few_shot_df.plot(kind='scatter', x='FewShotClassifier',  y='Recall',  color='blue', ax=ax)
    data_bayes_df.plot(kind='scatter', x='FewShotClassifier',  y='Precision',  color='red', title=title, ax=ax)
    data_bayes_df.plot(kind='scatter', x='FewShotClassifier',  y='Recall',  color='black', title=title, ax=ax)
    ax.set_ylabel("Class precision or recall")
    ax.legend(["Precision", "Recall", "N-Precision", "N-Recall"])
    plt.show()
      
    
def plotAccuracyModels(dataset, n_shot):
    title = dataset + " (n-shot=" + str(n_shot) + ")"

    model = "resnet18"

    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    #data_few_shot = resultDir + model + "_" + dataset + "_novelty_learn.txt"
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName
    data_few_shot = resultDir + model + "_" + dataset + "_novelty_test.txt"
    data_few_shot_df = pd.read_csv(data_few_shot)
    
    data_few_shot_18_df = data_few_shot_df[data_few_shot_df['Novelty'] == False]
    data_few_shot_18_df = data_few_shot_18_df.loc[data_few_shot_18_df['Shot'] == n_shot]
    
    data_novelty_df = data_few_shot_df[data_few_shot_df['Novelty'] == True]
    data_novelty_df = data_novelty_df.loc[data_novelty_df['Shot'] == n_shot]
    data_novelty_18_df = data_novelty_df.loc[data_novelty_df['Method'] == 'bayes']    
    
    model = "resnet34"
    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    #data_few_shot = resultDir + model + "_" + dataset + "_novelty_learn.txt"
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName
    data_few_shot = resultDir + model + "_" + dataset + "_novelty_test.txt"
    data_few_shot_df = pd.read_csv(data_few_shot)
   
    data_few_shot_34_df = data_few_shot_df[data_few_shot_df['Novelty'] == False]
    data_few_shot_34_df = data_few_shot_34_df.loc[data_few_shot_34_df['Shot'] == n_shot]

    data_novelty_df = data_few_shot_df[data_few_shot_df['Novelty'] == True]
    data_novelty_df = data_novelty_df.loc[data_novelty_df['Shot'] == n_shot]
    data_novelty_34_df = data_novelty_df.loc[data_novelty_df['Method'] == 'bayes']    
    
    model = "resnet50"
    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    #data_few_shot = resultDir + model + "_" + dataset + "_novelty_learn.txt"
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName
    data_few_shot = resultDir + model + "_" + dataset + "_novelty_test.txt"
    data_few_shot_df = pd.read_csv(data_few_shot)
   
    data_few_shot_50_df = data_few_shot_df[data_few_shot_df['Novelty'] == False]
    data_few_shot_50_df = data_few_shot_50_df.loc[data_few_shot_50_df['Shot'] == n_shot]

    data_novelty_df = data_few_shot_df[data_few_shot_df['Novelty'] == True]
    data_novelty_df = data_novelty_df.loc[data_novelty_df['Shot'] == n_shot]
    data_novelty_50_df = data_novelty_df.loc[data_novelty_df['Method'] == 'bayes']    
         
    ax = plt.gca()
    # Plot bayes vs. std
    data_few_shot_18_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='green', ax=ax)
    data_few_shot_34_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='blue', ax=ax)
    data_few_shot_50_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='red',  ax=ax)
    data_novelty_18_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='yellow', ax=ax)
    data_novelty_34_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='black', ax=ax)
    data_novelty_50_df.plot(kind='scatter', x='FewShotClassifier',  y='Accuracy',  color='orange', title=title, ax=ax)

    ax.set_ylabel("Accuracy")
    ax.legend(["FewShot 18", "FewShot 34", "FewShot 50", "Novelty 18", "Novelty 34", "Novelty 50"])
    plt.show()   
    
    
#%% MAIN
if __name__=='__main__':
    
    dataset = "Omniglot"
    plotAccuracy(dataset, model="resnet12", n_shot=1)
    plotAccuracy(dataset, model="resnet12", n_shot=5)
    plotPrecisionRecall(dataset, model="resnet12", n_shot=1)
    plotPrecisionRecall(dataset, model="resnet12", n_shot=5)

    dataset = "euMoths"
    plotAccuracy(dataset, model="resnet50", n_shot=1)
    plotAccuracy(dataset, model="resnet50", n_shot=5)
    plotPrecisionRecall(dataset, model="resnet50", n_shot=1)
    plotPrecisionRecall(dataset, model="resnet50", n_shot=5)
    
    plotAccuracyModels(dataset, n_shot=1)
    plotAccuracyModels(dataset, n_shot=5)

    dataset = "CUB"
    plotAccuracy(dataset, model="resnet18", n_shot=1)
    plotAccuracy(dataset, model="resnet18", n_shot=5)
    #plotAccuracy(dataset, model="resnet34", n_shot=1)
    #plotAccuracy(dataset, model="resnet34", n_shot=5)    
    plotPrecisionRecall(dataset, model="resnet18", n_shot=1)
    plotPrecisionRecall(dataset, model="resnet18", n_shot=5)
   
    plotAccuracyModels(dataset, n_shot=1)
    plotAccuracyModels(dataset, n_shot=5)

    dataset = "miniImagenet"
    plotAccuracy(dataset, model="resnet18", n_shot=1)
    plotAccuracy(dataset, model="resnet18", n_shot=5)
    #plotAccuracy(dataset, model="resnet34", n_shot=1)
    #plotAccuracy(dataset, model="resnet34", n_shot=5)    
    plotPrecisionRecall(dataset, model="resnet18", n_shot=1)
    plotPrecisionRecall(dataset, model="resnet18", n_shot=5)
    
    plotAccuracyModels(dataset, n_shot=1)
    plotAccuracyModels(dataset, n_shot=5)
 