# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:44:30 2023

@author: Kim Bjerge
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

n_shot=5

def plotBayesToleranceAvg(data_df, fewShotClassifier, title, limits, n_way=6, n_shot=5):
    
    data_df = data_df.loc[data_df['Novelty'] == True]
     
    # Header
    # Model,FewShotClassifier,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Percentage,Alpha,ModelName
    
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['Shot'] == n_shot]
    data_df = data_df.loc[data_df['Way'] == n_way]
    data_df = data_df.sort_values(by=['Percentage'])
    #print(data_df['Accuracy'].to_list())


    percentages = [97, 97, 98, 98.5, 99, 99.5, 100, 100.5, 101, 101.5, 102, 102.5, 103]
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for percentage in percentages:
        data_p_df = data_df.loc[data_df['Percentage'] == percentage]
        print(np.mean(data_p_df['Accuracy'].to_list()))
        accuracy.append(np.mean(data_p_df['Accuracy'].to_list()))
        precision.append(np.mean(data_p_df['Precision'].to_list()))
        recall.append(np.mean(data_p_df['Recall'].to_list()))
        f1.append(np.mean(data_p_df['F1'].to_list()))
        
    ax = plt.gca()
    plt.plot(percentages, accuracy, 'r')
    plt.plot(percentages, precision, 'b')
    plt.plot(percentages, recall, 'g')
    plt.plot(percentages, f1, 'k')

    plt.legend(["Accuracy", "Precision", "Recall", "F1"])
    plt.title(title)
    plt.ylabel('Score')
    plt.xlabel('Percentage')
    plt.ylim(limits) # Omniglot
    plt.show()
    

def plotBayesTolerance(data_df, fewShotClassifier, title, limits, n_way=6, n_shot=5):
    
    data_df = data_df.loc[data_df['Novelty'] == True]
     
    # Header
    # Model,FewShotClassifier,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Percentage,Alpha,ModelName
    
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['Shot'] == n_shot]
    data_df = data_df.loc[data_df['Way'] == n_way]
    data_df = data_df.sort_values(by=['Percentage'])
    print(data_df['Accuracy'].to_list())

    ax = plt.gca()
    data_df.plot(kind='line',
                x='Percentage',
                y='Accuracy',
                color='red', ax=ax)
    
    data_df.plot(kind='line',
                x='Percentage',
                y='Precision',
                color='blue', ax=ax)
    
    data_df.plot(kind='line',
                x='Percentage',
                y='Recall',
                color='green', ax=ax)
    
    data_df.plot(kind='line',
                x='Percentage',
                y='F1',
                color='black', ax=ax)
    
    # learned_distribution_resnet18_CUB.plot(kind='line',
    #                                        x='Shot',
    #                                        y='Std',
    #                                        color='green', ax=ax)
    plt.title(title)
    plt.ylabel('Score')
    plt.xlabel('Percentage')
    plt.ylim(limits) # Omniglot
    plt.show()
    

def plotsF1VsAlpha(data_dfs, colors, fewShotClassifier, title, limits, n_shot=n_shot):
    
    ax = plt.gca()
        
    legend = []
    idx = 0
    for data_df in data_dfs:

        data_df_novelty = data_df.loc[data_df['Novelty'] == True]
        data_df_novelty = data_df_novelty.loc[data_df_novelty['FewShotClassifier'] == fewShotClassifier]
        data_df_novelty = data_df_novelty.loc[data_df_novelty['Shot'] == n_shot]
        data_df_novelty['Way'] -= 1
        alpha = data_df['Alpha'][0]
        legend.append('Alpha ' + str(alpha))
        data_df_novelty.plot(kind='line',
                    x='Way',
                    y='F1',
                    color=colors[idx], ax=ax)
        idx += 1
        
    plt.title(title)
    plt.ylabel('F1-score')
    plt.xlabel('K-way')
    #plt.xlim(0, 40)
    plt.ylim(limits) # Omniglot
    plt.legend(legend)
    plt.show()

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

def plotScoresVsWays(data_df, fewShotClassifier, title, limits, Novelty=True, n_shot=n_shot):
    
    data_df_novelty = data_df.loc[data_df['Novelty'] == True]
    data_df = data_df.loc[data_df['Novelty'] == False]
    data_df_novelty = data_df_novelty.loc[data_df_novelty['FewShotClassifier'] == fewShotClassifier]
    data_df_novelty = data_df_novelty.loc[data_df_novelty['Shot'] == n_shot]
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['Shot'] == n_shot]
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
    plt.xlabel('K-way')
    plt.ylim(limits) # Omniglot
    #plt.xlim(0, 40)
    plt.legend(["FSL Acc.", "FSNL Acc.", "Precision", "Recall", "F1-Score"])
    plt.show()

def plotBayesAlphaOmniglot5Rand():

    data_df = pd.read_csv("./paperData/resnet12_Omniglot_novelty_bayes_test.txt")   
    plotBayesToleranceAvg(data_df, "Prototypical", "Omniglot learned TH (10-way)", (0.7, 1.0), n_shot=5, n_way=11)         

def plotBayesAlphaOmniglot():
    
    data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-bayes/resnet12_Omniglot_novelty_bayes_test_same.txt")   
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (5-way) (same test dataset)", (0.8, 1.0), n_shot=5, n_way=6)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 5-way (BD-CSPN)", (0.8, 1.0), n_shot=5, n_way=6)         

    data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-bayes/resnet12_Omniglot_novelty_bayes_test.txt")   
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (5-way)", (0.8, 1.0), n_shot=5, n_way=6)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 5-way (BD-CSPN)", (0.8, 1.0), n_shot=5, n_way=6)         
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (10-way)", (0.7, 1.0), n_shot=5, n_way=11)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 10-way (BD-CSPN)", (0.7, 1.0), n_shot=5, n_way=11)         
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (15-way)", (0.6, 1.0), n_shot=5, n_way=16)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 15-way (BD-CSPN)", (0.6, 1.0), n_shot=5, n_way=16)         
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (20-way)", (0.55, 1.0), n_shot=5, n_way=21)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 20-way (BD-CSPN)", (0.55, 1.0), n_shot=5, n_way=21)         
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (25-way)", (0.5, 1.0), n_shot=5, n_way=26)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 25-way (BD-CSPN)", (0.5, 1.0), n_shot=5, n_way=26)         
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (30-way)", (0.5, 1.0), n_shot=5, n_way=31)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 30-way (BD-CSPN)", (0.5, 1.0), n_shot=5, n_way=31)         
    plotBayesTolerance(data_df, "Prototypical", "Omniglot learned TH (35-way)", (0.5, 1.0), n_shot=5, n_way=36)         
    #plotBayesTolerance(data_df, "BD-CSPN", "Omniglot learned TH 35-way (BD-CSPN)", (0.5, 1.0), n_shot=5, n_way=36)         
    #data_df = pd.read_csv("./modelsOmniglotAdvStd1/resnet12_Omniglot_novelty_test.txt")
    #data_df = pd.read_csv("./modelsOmniglotAdvStd3/resnet12_Omniglot_novelty_test_GPU.txt")
    #data_df = pd.read_csv("./modelsOmniglotAdvStd3/resnet12_Omniglot_novelty_test.txt")

    #data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-5w/resnet12_Omniglot_novelty_test_GPU.txt")
    #data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-5w/resnet12_Omniglot_novelty_test_CPU.txt")

    data_df = pd.read_csv("./modelsOmniglotAdvStd4/results-5w/resnet12_Omniglot_novelty_test_CPU.txt")
    plotScorsVsAlpha(data_df, "Prototypical", "Omniglot Novelty vs. Alpha" , (0.8, 1.0), n_shot=5)
    plotScorsVsAlpha(data_df, "BD-CSPN", "Omniglot Novelty vs. Alpha (BD-CSPN)", (0.8, 1.0), n_shot=5)

    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-5w/resnet12_Omniglot_novelty_test_GPU.txt")
    #plotScorsVsAlpha(data_df, "Prototypical", "Omniglot R2 Prototypical", (0.8, 1.0), n_shot=5)
    #plotScorsVsAlpha(data_df, "BD-CSPN", "Omniglot R2 BD-CSPN", (0.8, 1.0), n_shot=5)

def plotWaysOmniglot(fewShotClassifier):

    data_df_0 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_0_novelty_ways_test.txt")
    plotScoresVsWays(data_df_0, fewShotClassifier, "Omniglot (Alpha=0.0, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_1 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_1_novelty_ways_test.txt")
    plotScoresVsWays(data_df_1, fewShotClassifier, "Omniglot (Alpha=0.1, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_2 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_2_novelty_ways_test.txt")
    plotScoresVsWays(data_df_2, fewShotClassifier, "Omniglot (Alpha=0.2, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_3 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_3_novelty_ways_test.txt")
    #plotScoresVsWays(data_df_3, fewShotClassifier, "Omniglot (Alpha=0.3, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_4 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_4_novelty_ways_test.txt")
    #plotScoresVsWays(data_df_4, fewShotClassifier, "Omniglot (Alpha=0.4, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_5 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_5_novelty_ways_test.txt")
    #plotScoresVsWays(data_df_5, fewShotClassifier, "Omniglot (Alpha=0.5, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_6 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_6_novelty_ways_test.txt")
    #plotScoresVsWays(data_df_6, fewShotClassifier, "Omniglot (Alpha=0.6, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_7 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_7_novelty_ways_test.txt")
    #plotScoresVsWays(data_df_7, fewShotClassifier, "Omniglot (Alpha=0.7, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_8 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_8_novelty_ways_test.txt")
    plotScoresVsWays(data_df_8, fewShotClassifier, "Omniglot (Alpha=0.8)", (0.5, 1.0))
    #plotScoresVsWays(data_df_8, fewShotClassifier, "Omniglot (Alpha=0.8, " + fewShotClassifier + ")", (0.5, 1.0))
    data_df_9 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_9_novelty_ways_test.txt")
    plotScoresVsWays(data_df_9, fewShotClassifier, "Omniglot (Alpha=0.9, " + fewShotClassifier + ")" , (0.5, 1.0))
    data_df_10 = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_10_novelty_ways_test.txt")
    plotScoresVsWays(data_df_10, fewShotClassifier, "Omniglot (Alpha=1.0, " + fewShotClassifier + ")", (0.5, 1.0))
    
    colors = ['black', 'blue', 'green', 'brown', 'orange', 'red']
    data_dfs = [data_df_0, data_df_1, data_df_2, data_df_8, data_df_9, data_df_10]
    plotsF1VsAlpha(data_dfs, colors, fewShotClassifier, "Omniglot Novelty F1-score", (0.5, 1.0))

    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-Nw/resnet12_Omniglot_8_novelty_ways_test.txt")
    #plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R2 0.8 " + fewShotClassifier, (0.6, 1.0))
    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-Nw/resnet12_Omniglot_9_novelty_ways_test.txt")
    #plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R2 0.9 " + fewShotClassifier, (0.6, 1.0))
    #data_df = pd.read_csv("./modelsOmniglotAdvStd4_1/results-Nw/resnet12_Omniglot_10_novelty_ways_test.txt")
    #plotScoresVsWays(data_df, fewShotClassifier, "Omniglot R2 1.0 " + fewShotClassifier, (0.6, 1.0))    

# def plotWaysEUMoths(fewShotClassifier):

#     data_df = pd.read_csv("./modelsFinalPreAdv/results_Nw/resnet18_euMoths_5_novelty_ways_test.txt")
#     plotScoresVsWays(data_df, fewShotClassifier, "EU moths (Alpha=0.5)", (0.1, 1.0))
#     #plotScoresVsWays(data_df, fewShotClassifier, "EU moths (Alpha=0.5 " + fewShotClassifier + ")", (0.1, 1.0))
#     data_df = pd.read_csv("./modelsFinalPreAdv/results_Nw/resnet18_euMoths_10_novelty_ways_test.txt")
#     plotScoresVsWays(data_df, fewShotClassifier, "EU moths (Alpha=1.0 )", (0.1, 1.0))
#     #plotScoresVsWays(data_df, fewShotClassifier, "EU moths (Alpha=1.0 " + fewShotClassifier + ")", (0.1, 1.0))

def plotWaysMultiOmniglot(fewShotClassifier):

    data_df_0 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/resnet12_Omniglot_0_novelty_ways_test.txt")
    plotScoresVsWays(data_df_0, fewShotClassifier, "Omniglot (Alpha=0.0, " + fewShotClassifier + ")", (0, 1.0))
    data_df_1 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/resnet12_Omniglot_1_novelty_ways_test.txt")
    plotScoresVsWays(data_df_1, fewShotClassifier, "Omniglot (Alpha=0.1, " + fewShotClassifier + ")", (0, 1.0))
    data_df_2 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/resnet12_Omniglot_2_novelty_ways_test.txt")
    plotScoresVsWays(data_df_2, fewShotClassifier, "Omniglot (Alpha=0.2, " + fewShotClassifier + ")", (0, 1.0))
    data_df_5 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/train20ways/resnet12_Omniglot_5_novelty_ways_test.txt")
    #data_df_5 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/resnet12_Omniglot_5_novelty_ways_test.txt")
    plotScoresVsWays(data_df_5, fewShotClassifier, "Omniglot (Alpha=0.5, " + fewShotClassifier + ")", (0, 1.0))
    data_df_8 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/resnet12_Omniglot_8_novelty_ways_test.txt")
    plotScoresVsWays(data_df_8, fewShotClassifier, "Omniglot (Alpha=0.8, " + fewShotClassifier + ")", (0, 1.0))
    data_df_9 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/resnet12_Omniglot_9_novelty_ways_test.txt")
    plotScoresVsWays(data_df_9, fewShotClassifier, "Omniglot (Alpha=0.9, " + fewShotClassifier + ")" , (0, 1.0))
    data_df_10 = pd.read_csv("./modelsOmniglotAdvMulti4/results-Nw/resnet12_Omniglot_10_novelty_ways_test.txt")
    plotScoresVsWays(data_df_10, fewShotClassifier, "Omniglot (Alpha=1.0, " + fewShotClassifier + ")", (0, 1.0))
    
    colors = ['black', 'blue', 'green', 'brown', 'orange', 'red', 'magneta']
    data_dfs = [data_df_0, data_df_1, data_df_2, data_df_5, data_df_8,  data_df_10]
    plotsF1VsAlpha(data_dfs, colors, fewShotClassifier, "Omniglot Novelty F1-score", (0, 1.0))


def plotWaysMiniImagenet(fewShotClassifier):

    data_dfs = []
    for alpha in range(11):
        if alpha in [3, 4, 5, 6, 7]:
            continue
        resultFile = "./modelsAlphaMiniImageNet/results-Nw/resnet18_miniImagenet_" + str(alpha) + "_novelty_ways_test.txt"
        data_df = pd.read_csv(resultFile)
        alpha /= 10
        plotScoresVsWays(data_df, fewShotClassifier, f"miniImageNet (Alpha={alpha:.1f})", (0.0, 1.0))
        data_dfs.append(data_df)
 
    colors = ['black', 'blue', 'green', 'brown', 'orange', 'red']
    plotsF1VsAlpha(data_dfs, colors, fewShotClassifier, "miniImageNet Novelty F1-score", (0.0, 1.0))


def plotWaysEUMoths(fewShotClassifier):

    data_dfs = []
    for alpha in range(11):
        if alpha in [3, 4, 5, 6, 7]:
            continue
        resultFile = "./modelsAlphaEUMoths/results-Nw/resnet18_euMoths_" + str(alpha) + "_novelty_ways_test.txt"
        data_df = pd.read_csv(resultFile)
        alpha /= 10
        plotScoresVsWays(data_df, fewShotClassifier, f"EU Moths (Alpha={alpha:.1f})", (0.2, 1.0))
        data_dfs.append(data_df)
 
    colors = ['black', 'blue', 'green', 'brown', 'orange', 'red', 'purple']
    plotsF1VsAlpha(data_dfs, colors, fewShotClassifier, "", (0.2, 1.0))       


def plotWaysMultiEUMoths(fewShotClassifier):

    data_dfs = []
    for alpha in range(11):
        if alpha in [3, 4, 5, 6, 7]:
            continue
        resultFile = "./modelsAlphaEUMothsMulti/results-Nw/resnet18_euMoths_" + str(alpha) + "_novelty_ways_test.txt"
        data_df = pd.read_csv(resultFile)
        alpha /= 10
        plotScoresVsWays(data_df, fewShotClassifier, f"EU Moths (Alpha={alpha:.1f})", (0.0, 1.0))
        data_dfs.append(data_df)
 
    colors = ['black', 'blue', 'green', 'brown', 'orange', 'red', 'purple']
    plotsF1VsAlpha(data_dfs, colors, fewShotClassifier, "", (0.0, 1.0))   
    
def plotWaysCUBs(fewShotClassifier):

    data_dfs = []
    for alpha in range(11):
        if alpha in [3, 4, 5, 6, 7]:
            continue
        resultFile = "./modelsAlphaCUB/results-Nw/resnet18_CUB_" + str(alpha) + "_novelty_ways_test.txt"
        data_df = pd.read_csv(resultFile)
        alpha /= 10
        plotScoresVsWays(data_df, fewShotClassifier, f"CUB (Alpha={alpha:.1f})", (0.2, 1.0))
        data_dfs.append(data_df)
 
    colors = ['black', 'blue', 'green', 'brown', 'orange', 'red', 'purple']
    plotsF1VsAlpha(data_dfs, colors, fewShotClassifier, "", (0.2, 1.0))       


#%% MAIN
if __name__=='__main__':

    
    #plotBayesAlphaOmniglot()
    #plotBayesAlphaOmniglot5Rand()

    fewShotClassifier = "Prototypical"
    #fewShotClassifier = "BD-CSPN"
    plotWaysMultiOmniglot(fewShotClassifier)
    #plotWaysOmniglot(fewShotClassifier)    

    #plotWaysEUMoths(fewShotClassifier)
    #plotWaysMultiEUMoths(fewShotClassifier)
    #fewShotClassifier = "BD-CSPN"
    #plotWaysEUMoths(fewShotClassifier)
    #plotWaysMiniImagenet(fewShotClassifier)

    #plotWaysCUBs(fewShotClassifier)
    