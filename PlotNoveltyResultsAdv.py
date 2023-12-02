# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:44:30 2023

@author: Kim Bjerge
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotScorsVsAlpha(data_df, fewShotClassifier, title, limits, n_shot=5):
    
    data_df = data_df.loc[data_df['Novelty'] == True]
     
    # Header
    # Model,FewShotClassifier,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName
    
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['Shot'] == n_shot]
    data_df = data_df.sort_values(by=['Alpha'])
    print(data_df['Accuracy'].to_list())

    ax = plt.gca()
    data_df.plot(kind='line',
                x='Alpha',
                y='Accuracy',
                color='red', ax=ax)
    
    data_df.plot(kind='line',
                x='Alpha',
                y='Precision',
                color='blue', ax=ax)
    
    data_df.plot(kind='line',
                x='Alpha',
                y='Recall',
                color='green', ax=ax)
    
    data_df.plot(kind='line',
                x='Alpha',
                y='F1',
                color='black', ax=ax)
    
    # learned_distribution_resnet18_CUB.plot(kind='line',
    #                                        x='Shot',
    #                                        y='Std',
    #                                        color='green', ax=ax)
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(limits) # Omniglot
    plt.show()

def plotScoresVsWays(data_df, fewShotClassifier, title, limits, Novelty=True):
    
    data_df_novelty = data_df.loc[data_df['Novelty'] == True]
    data_df = data_df.loc[data_df['Novelty'] == False]
    data_df_novelty = data_df_novelty.loc[data_df_novelty['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df_novelty['Way'] -= 1
    
    ax = plt.gca()
    data_df.plot(kind='line',
                x='Way',
                y='Accuracy',
                color='orange', ax=ax)
    
    data_df_novelty.plot(kind='line',
                x='Way',
                y='Accuracy',
                color='red', ax=ax)
 
    if Novelty:
        data_df_novelty.plot(kind='line',
                    x='Way',
                    y='Precision',
                    color='blue', ax=ax)
        
        data_df_novelty.plot(kind='line',
                    x='Way',
                    y='Recall',
                    color='green', ax=ax)
        
        data_df_novelty.plot(kind='line',
                    x='Way',
                    y='F1',
                    color='black', ax=ax)
    else:
        data_df.plot(kind='line',
                    x='Way',
                    y='Precision',
                    color='blue', ax=ax)
        
        data_df.plot(kind='line',
                    x='Way',
                    y='Recall',
                    color='green', ax=ax)
        
        data_df.plot(kind='line',
                    x='Way',
                    y='F1',
                    color='black', ax=ax)
    
    # learned_distribution_resnet18_CUB.plot(kind='line',
    #                                        x='Shot',
    #                                        y='Std',
    #                                        color='green', ax=ax)
    #plt.title('Omniglot dataset ')
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(limits) # Omniglot
    plt.legend(["AccFewShot", "AccNovelty", "Precision", "Recall", "F1-Score"])
    plt.show()
    

#%% MAIN
if __name__=='__main__':
    
    
    
    #data_df = pd.read_csv("./modelsOmniglotAdvStd1/resnet12_Omniglot_novelty_test.txt")
    #data_df = pd.read_csv("./modelsOmniglotAdvStd3/resnet12_Omniglot_novelty_test_GPU.txt")
    #data_df = pd.read_csv("./modelsOmniglotAdvStd3/resnet12_Omniglot_novelty_test.txt")

    #data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-5w/resnet12_Omniglot_novelty_test_GPU.txt")
    #data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-5w/resnet12_Omniglot_novelty_test_CPU.txt")

    data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-5w/resnet12_Omniglot_novelty_test_CPU.txt")
    plotScorsVsAlpha(data_df, "Prototypical", "Omniglot R1 Prototypical", (0.8, 1.0), n_shot=5)
    plotScorsVsAlpha(data_df, "BD-CSPN", "Omniglot R1 BD-CSPN", (0.8, 1.0), n_shot=5)

    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-5w/resnet12_Omniglot_novelty_test_GPU.txt")
    #plotScorsVsAlpha(data_df, "Prototypical", "Omniglot R2 Prototypical", (0.8, 1.0), n_shot=5)
    #plotScorsVsAlpha(data_df, "BD-CSPN", "Omniglot R2 BD-CSPN", (0.8, 1.0), n_shot=5)
    
    fewShotClassifier = "Prototypical"
    #fewShotClassifier = "BD-CSPN"

    data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_10_novelty_ways_test.txt")
    plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R1 1.0 " + fewShotClassifier, (0.6, 1.0))
    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-Nw/resnet12_Omniglot_10_novelty_ways_test.txt")
    #plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R2 1.0 " + fewShotClassifier, (0.6, 1.0))
    data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_9_novelty_ways_test.txt")
    plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R1 0.9 " + fewShotClassifier, (0.6, 1.0))
    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-Nw/resnet12_Omniglot_9_novelty_ways_test.txt")
    #plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R2 0.9 " + fewShotClassifier, (0.6, 1.0))
    data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_8_novelty_ways_test.txt")
    plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R1 0.8 " + fewShotClassifier, (0.6, 1.0))
    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-Nw/resnet12_Omniglot_8_novelty_ways_test.txt")
    #plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R2 0.8 " + fewShotClassifier, (0.6, 1.0))
    
    data_df = pd.read_csv("./modelsFinalPreAdv/results_Nw/resnet18_euMoths_5_novelty_ways_test.txt")
    plotScoresVsWays(data_df, fewShotClassifier, "EU moths R1 0.5 " + fewShotClassifier, (0.1, 1.0))
    data_df = pd.read_csv("./modelsFinalPreAdv/results_Nw/resnet18_euMoths_10_novelty_ways_test.txt")
    plotScoresVsWays(data_df, fewShotClassifier, "EU moths R1 1.0 " + fewShotClassifier, (0.1, 1.0))