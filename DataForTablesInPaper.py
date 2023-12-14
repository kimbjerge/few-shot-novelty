# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:36:20 2023

@author: Kim Bjerge
"""
import pandas as pd

def dataForTable(data_df_novelty, data_df_learn, fewShotClassifier, title, ways=[5, 10, 20, 30], n_shot=5):
    
    # Header learn
    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    #data_df_learn = data_df_novelty.loc[data_df_novelty['Novelty'] == False]
    data_df_learn = data_df_learn.loc[data_df_learn['FewShotClassifier'] == fewShotClassifier]
    data_df_learn = data_df_learn.loc[data_df_learn['Shot'] == n_shot]
    data_df_learn = data_df_learn.sort_values(by=['Way'])
    print(data_df_learn['Accuracy'].to_list())
    print(data_df_learn['Way'].to_list())

    # Header novelty
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName
    data_df = data_df_novelty.loc[data_df_novelty['Novelty'] == True]
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['Shot'] == n_shot]
    data_df = data_df.sort_values(by=['Way'])
    print(data_df['Accuracy'].to_list())
    print(data_df['Way'].to_list())
    
    fewShotAcc = []
    noveltyAcc = []
    precision = []
    recall = []
    F1 = []
    #for n_way in [5, 10, 20, 30]:
    #for n_way in [5, 10, 20, 30, 40]:
    for n_way in ways:
        fewShotAcc.append(data_df_learn.loc[data_df_learn['Way'] == n_way]["Accuracy"].iloc[0])
        data_df_way = data_df.loc[data_df['Way'] == n_way-1] # Novelty 6-way = 5-way + 1-novel
        noveltyAcc.append(data_df_way["Accuracy"].iloc[0])
        precision.append(data_df_way["Precision"].iloc[0])
        recall.append(data_df_way["Recall"].iloc[0])
        F1.append(data_df_way["F1"].iloc[0])
    
    if len(ways) == 4:
        line = ""
        line += f"FewShot   & {fewShotAcc[0]:.3f} & {fewShotAcc[1]:.3f} & {fewShotAcc[2]:.3f} & {fewShotAcc[3]:.3f} \\\\ \n"
        line += f"Novelty   & {noveltyAcc[0]:.3f} & {noveltyAcc[1]:.3f} & {noveltyAcc[2]:.3f} & {noveltyAcc[3]:.3f} \\\\ \n"
        line += f"Precision & {precision[0]:.3f} & {precision[1]:.3f} & {precision[2]:.3f} & {precision[3]:.3f} \\\\ \n"
        line += f"Recall    & {recall[0]:.3f} & {recall[1]:.3f} & {recall[2]:.3f} & {recall[3]:.3f} \\\\ \n"
        line += f"F1-score  & {F1[0]:.3f} & {F1[1]:.3f} & {F1[2]:.3f} & {F1[3]:.3f} \\\\ \n"
    else:    
        line = ""
        line += f"FewShot   & {fewShotAcc[0]:.3f} & {fewShotAcc[1]:.3f} & {fewShotAcc[2]:.3f} & {fewShotAcc[3]:.3f} & {fewShotAcc[4]:.3f} \\\\ \n"
        line += f"Novelty   & {noveltyAcc[0]:.3f} & {noveltyAcc[1]:.3f} & {noveltyAcc[2]:.3f} & {noveltyAcc[3]:.3f} & {noveltyAcc[4]:.3f} \\\\ \n"
        line += f"Precision & {precision[0]:.3f} & {precision[1]:.3f} & {precision[2]:.3f} & {precision[3]:.3f} & {precision[4]:.3f} \\\\ \n"
        line += f"Recall    & {recall[0]:.3f} & {recall[1]:.3f} & {recall[2]:.3f} & {recall[3]:.3f} & {recall[4]:.3f} \\\\ \n"
        line += f"F1-score  & {F1[0]:.3f} & {F1[1]:.3f} & {F1[2]:.3f} & {F1[3]:.3f} & {F1[4]:.3f} \\\\ \n"
    
    print(title)
    print(line)
    
def dataForTable5w(data_df, fewShotClassifier, novelty):  
    
    for n_shot in [1, 5]:
        
        data_df_n = data_df.loc[data_df['Novelty'] == novelty]
        data_df_n = data_df_n.loc[data_df_n['FewShotClassifier'] == fewShotClassifier]
        data_df_n = data_df_n.loc[data_df_n['Shot'] == n_shot]
        data_df_n = data_df_n.sort_values(by=['Way'])
        print(f"Accuracy with novelty {novelty:d} for {fewShotClassifier:s} with N-shot {n_shot:d}", 
              data_df_n['Accuracy'].to_list(),
              data_df_n['Precision'].to_list(),
              data_df_n['Recall'].to_list(),
              data_df_n['F1'].to_list())
    
#%% MAIN
if __name__=='__main__':
    
    fewShotClassifier = "Prototypical"
    #fewShotClassifier = "BD-CSPN"

    """
    data_df_10_learn = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_10_novelty_ways_learn.txt")    
    data_df_10_novelty = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_10_novelty_ways_test.txt")
    dataForTable(data_df_10_novelty, data_df_10_learn, fewShotClassifier, "Omniglot (Alpha=1.0)")

    data_df_0_learn = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_0_novelty_ways_learn.txt")    
    data_df_0_novelty = pd.read_csv("./modelsOmniglotAdvStd4/results-Nw/resnet12_Omniglot_0_novelty_ways_test.txt")
    dataForTable(data_df_0_novelty, data_df_0_learn, fewShotClassifier, "Omniglot (Alpha=0.0)")
    """

    data_df_10_learn = pd.read_csv("./modelsAlphaEUMoths/results-Nw/resnet18_euMoths_10_novelty_ways_learn.txt")    
    data_df_10_novelty = pd.read_csv("./modelsAlphaEUMoths/results-Nw/resnet18_euMoths_10_novelty_ways_test.txt")
    dataForTable(data_df_10_novelty, data_df_10_learn, fewShotClassifier, "EU Moths (Alpha=1.0)", ways=[5, 10, 20, 30, 40])

    data_df = pd.read_csv("./modelsAlphaEUMoths/results-5w/resnet18_euMoths_novelty_test.txt") 
    print("EU Moths")
    dataForTable5w(data_df, fewShotClassifier, novelty=True)
    dataForTable5w(data_df, fewShotClassifier, novelty=False)
 
    data_df = pd.read_csv("./modelsAlphaCUB/results-5w/resnet18_CUB_novelty_test.txt") 
    print("CUB")
    dataForTable5w(data_df, fewShotClassifier, novelty=True)
    dataForTable5w(data_df, fewShotClassifier, novelty=False)   
    
    data_df_10_learn = pd.read_csv("./modelsAlphaCUB/results-Nw/resnet18_CUB_10_novelty_ways_learn.txt")    
    data_df_10_novelty = pd.read_csv("./modelsAlphaCUB/results-Nw/resnet18_CUB_10_novelty_ways_test.txt")
    dataForTable(data_df_10_novelty, data_df_10_learn, fewShotClassifier, "CUB (Alpha=1.0)", ways=[5, 10, 20, 28])
    
    data_df_10_learn = pd.read_csv("./modelsAlphaMiniImageNet/results-Nw/resnet18_miniImagenet_1_novelty_ways_learn.txt")    
    data_df_10_novelty = pd.read_csv("./modelsAlphaMiniImageNet/results-Nw/resnet18_miniImagenet_1_novelty_ways_test.txt")
    dataForTable(data_df_10_novelty, data_df_10_learn, fewShotClassifier, "miniImageNet (Alpha=0.1)", ways=[5, 10, 14, 18])
    
    #fewShotClassifier = "BD-CSPN"
    data_df = pd.read_csv("./modelsAlphaMiniImageNet/results-5w/resnet18_miniImagenet_novelty_test.txt") 
    print("miniImageNet")
    dataForTable5w(data_df, fewShotClassifier, novelty=True)
    dataForTable5w(data_df, fewShotClassifier, novelty=False)   