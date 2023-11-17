# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:44:30 2023

@author: Kim Bjerge
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% MAIN
if __name__=='__main__':
    
    
    n_shot = 5
    fewShotClassifier = "Prototypical"
    fewShotClassifier = "BD-CSPN"
    data_df = pd.read_csv("./modelsOmniglotAdvStd1/resnet12_Omniglot_novelty_test.txt")
    data_df = pd.read_csv("./modelsOmniglotAdvStd3/resnet12_Omniglot_novelty_test.txt")
     
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
    plt.title('Omniglot dataset ')
    plt.ylabel('Score')
    plt.ylim(0.8, 1.0) # Omniglot
    plt.show()