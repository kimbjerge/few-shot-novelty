# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:45:58 2023

@author: Kim Bjerge
"""

import matplotlib.pyplot as plt
import pandas as pd

resultDir = "./result/testT200/" # 200*6 queries 

def plotAccuracy(dataset, model, n_shot):
    title = dataset + " " + model + " (n-shot=" + str(n_shot) + ")"
    
    # Model,FewShotClassifier,Way,Shot,Query,Accuracy,Method,Threshold
    data_few_shot = resultDir + dataset + "_few_shot.txt"
    # Model,FewShotClassifier,Way,Shot,Query,Accuracy,Method,Threshold
    data_novelty = resultDir + dataset + "_novelty.txt"
    
    data_few_shot_df = pd.read_csv(data_few_shot)
    data_few_shot_df = data_few_shot_df[data_few_shot_df['Model'] == model]
    data_novelty_df = pd.read_csv(data_novelty)
    data_novelty_df = data_novelty_df[data_novelty_df['Model'] == model]
    
    data_df = data_novelty_df.loc[data_novelty_df['Shot'] == n_shot]
    data_few_shot_df = data_few_shot_df.loc[data_few_shot_df['Shot'] == n_shot]
    data_bayes_df = data_df.loc[data_df['Method'] == 'bayes']
    data_std_df = data_df.loc[data_df['Method'] == 'std']
 
    ax = plt.gca()
    # Plot bayes vs. std
    data_few_shot_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='green', ax=ax)
    data_std_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='blue', ax=ax)
    data_bayes_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='red', title=title, ax=ax)
    ax.set_ylabel("Accuracy")
    ax.legend(["FewShot", "NoveltyStd", "NoveltyBayes"])
    plt.show()
      
    
def plotAccuracyModels(dataset, n_shot):
    title = dataset + " (n-shot=" + str(n_shot) + ")"
    
    # Model,FewShotClassifier,Way,Shot,Query,Accuracy,Method,Threshold
    data_few_shot = resultDir + dataset + "_few_shot.txt"
    # Model,FewShotClassifier,Way,Shot,Query,Accuracy,Method,Threshold
    data_novelty = resultDir + dataset + "_novelty.txt"
    
    data_few_shot_df = pd.read_csv(data_few_shot)
    data_few_shot_df = data_few_shot_df.loc[data_few_shot_df['Shot'] == n_shot]

    data_few_shot_18_df = data_few_shot_df[data_few_shot_df['Model'] == 'resnet18']
    data_few_shot_34_df = data_few_shot_df[data_few_shot_df['Model'] == 'resnet34']
    data_few_shot_50_df = data_few_shot_df[data_few_shot_df['Model'] == 'resnet50']
    
    data_novelty_df = pd.read_csv(data_novelty)
    data_novelty_df = data_novelty_df.loc[data_novelty_df['Shot'] == n_shot]
    data_novelty_df = data_novelty_df.loc[data_novelty_df['Method'] == 'bayes']
    
    data_novelty_18_df = data_novelty_df[data_novelty_df['Model'] == 'resnet18']
    data_novelty_34_df = data_novelty_df[data_novelty_df['Model'] == 'resnet34']
    data_novelty_50_df = data_novelty_df[data_novelty_df['Model'] == 'resnet50']
    
 
    ax = plt.gca()
    # Plot bayes vs. std
    data_few_shot_18_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='green', ax=ax)
    data_few_shot_34_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='blue', ax=ax)
    data_few_shot_50_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='red',  ax=ax)
    data_novelty_18_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='yellow', ax=ax)
    data_novelty_34_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='black', ax=ax)
    data_novelty_50_df.plot(kind='line', x='FewShotClassifier',  y='Accuracy',  color='orange', title=title, ax=ax)

    ax.set_ylabel("Accuracy")
    ax.legend(["FewShot 18", "FewShot 34", "FewShot 50", "Novelty 18", "Novelty 34", "Novelty 50"])
    plt.show()   
    
    
#%% MAIN
if __name__=='__main__':
    
    dataset = "Omniglot"
    plotAccuracy(dataset, model="resnet12", n_shot=1)
    plotAccuracy(dataset, model="resnet12", n_shot=5)

    dataset = "euMoths"
    plotAccuracy(dataset, model="resnet18", n_shot=1)
    plotAccuracy(dataset, model="resnet18", n_shot=5)
    
    plotAccuracyModels(dataset, n_shot=1)
    plotAccuracyModels(dataset, n_shot=5)

    dataset = "CUB"
    plotAccuracy(dataset, model="resnet18", n_shot=1)
    plotAccuracy(dataset, model="resnet18", n_shot=5)
    plotAccuracy(dataset, model="resnet34", n_shot=1)
    plotAccuracy(dataset, model="resnet34", n_shot=5)    
    
    plotAccuracyModels(dataset, n_shot=1)
    plotAccuracyModels(dataset, n_shot=5)

    dataset = "miniImagenet"
    plotAccuracy(dataset, model="resnet18", n_shot=1)
    plotAccuracy(dataset, model="resnet18", n_shot=5)
    plotAccuracy(dataset, model="resnet34", n_shot=1)
    plotAccuracy(dataset, model="resnet34", n_shot=5)    
    
    plotAccuracyModels(dataset, n_shot=1)
    plotAccuracyModels(dataset, n_shot=5)
 