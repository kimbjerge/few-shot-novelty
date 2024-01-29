# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:36:20 2023

@author: Kim Bjerge
"""
import pandas as pd
import numpy as np

def dataForTableAvg1(data_df_novelty, fewShotClassifier, title, ways=[5, 10, 20, 30], n_shot=5):
    
    #print("dataForTableAvg", fewShotClassifier, ways, n_shot)
    # Header learn
    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    data_df_learn = data_df_novelty.loc[data_df_novelty['Novelty'] == False]
    data_df_learn = data_df_learn.loc[data_df_learn['FewShotClassifier'] == fewShotClassifier]
    data_df_learn = data_df_learn.loc[data_df_learn['Shot'] == n_shot]
    data_df_learn = data_df_learn.sort_values(by=['Way'])
    #print(data_df_learn['Accuracy'].to_list())
    #print("FSL K-way", data_df_learn['Way'].to_list())

    # Header novelty
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,Seed,ModelName
    data_df = data_df_novelty.loc[data_df_novelty['Novelty'] == True]
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['Shot'] == n_shot]
    data_df = data_df.sort_values(by=['Way'])
    #print(data_df['Accuracy'].to_list())
    #print("FSNL K-way", data_df['Way'].to_list())
    
    fewShotAcc = []
    noveltyAcc = []
    precision = []
    recall = []
    F1 = []
    #for n_way in [5, 10, 20, 30]:
    #for n_way in [5, 10, 20, 30, 40]:
    for n_way in ways:
        accuracyFSL = data_df_learn.loc[data_df_learn['Way'] == n_way]["Accuracy"].to_list()
        fewShotAcc.append([np.mean(accuracyFSL), np.std(accuracyFSL)])
        data_df_way = data_df.loc[data_df['Way'] == n_way+1] # Novelty 6-way = 5-way + 1-novel
        accuracyFSNL = data_df_way["Accuracy"].to_list()
        noveltyAcc.append([np.mean(accuracyFSNL), np.std(accuracyFSNL)])
        precisionFSNL = data_df_way["Precision"].to_list()
        precision.append([np.mean(precisionFSNL), np.std(precisionFSNL)])
        recallFSNL = data_df_way["Recall"].to_list()
        recall.append([np.mean(recallFSNL), np.std(recallFSNL)])
        F1FSNL = data_df_way["F1"].to_list()
        F1.append([np.mean(F1FSNL), np.std(F1FSNL)])
    
    if len(ways) == 4:
        line = ""
        line += f"Acc. (FSL)  & {fewShotAcc[0][0]:.3f} \\footnotesize({fewShotAcc[0][1]:.3f}) & {fewShotAcc[1][0]:.3f} \\footnotesize({fewShotAcc[1][1]:.3f}) & "
        line +=             f"{fewShotAcc[2][0]:.3f} \\footnotesize({fewShotAcc[2][1]:.3f}) & {fewShotAcc[3][0]:.3f} \\footnotesize({fewShotAcc[3][1]:.3f}) \\\\ \n"
        line += f"Acc. (FSNL) & {noveltyAcc[0][0]:.3f} \\footnotesize({noveltyAcc[0][1]:.3f}) & {noveltyAcc[1][0]:.3f} \\footnotesize({noveltyAcc[1][1]:.3f}) & "
        line +=             f"{noveltyAcc[2][0]:.3f} \\footnotesize({noveltyAcc[2][1]:.3f}) & {noveltyAcc[3][0]:.3f} \\footnotesize({noveltyAcc[3][1]:.3f}) \\\\ \n"
        line += f"Precision   & {precision[0][0]:.3f} \\footnotesize({precision[0][1]:.3f}) & {precision[1][0]:.3f} \\footnotesize({precision[1][1]:.3f}) & "
        line +=             f"{precision[2][0]:.3f} \\footnotesize({precision[2][1]:.3f}) & {precision[3][0]:.3f} \\footnotesize({precision[3][1]:.3f}) \\\\ \n"
        line += f"Recall      & {recall[0][0]:.3f} \\footnotesize({recall[0][1]:.3f}) & {recall[1][0]:.3f} \\footnotesize({recall[1][1]:.3f}) & "
        line +=             f"{recall[2][0]:.3f} \\footnotesize({recall[2][1]:.3f}) & {recall[3][0]:.3f} \\footnotesize({recall[3][1]:.3f}) \\\\ \n"
        line += f"F1-score    & {F1[0][0]:.3f} \\footnotesize({F1[0][1]:.3f}) & {F1[1][0]:.3f} \\footnotesize({F1[1][1]:.3f}) & "
        line +=             f"{F1[2][0]:.3f} \\footnotesize({F1[2][1]:.3f}) & {F1[3][0]:.3f} \\footnotesize({F1[3][1]:.3f}) \\\\ \n"
    else:    
        line = ""
        line += f"Acc. (FSL)  & {fewShotAcc[0][0]:.3f} \\footnotesize({fewShotAcc[0][1]:.3f}) & {fewShotAcc[1][0]:.3f} \\footnotesize({fewShotAcc[1][1]:.3f}) & "
        line +=             f"{fewShotAcc[2][0]:.3f} \\footnotesize({fewShotAcc[2][1]:.3f}) & {fewShotAcc[3][0]:.3f} \\footnotesize({fewShotAcc[3][1]:.3f}) & {fewShotAcc[4][0]:.3f} \\footnotesize({fewShotAcc[4][1]:.3f}) \\\\ \n"
        line += f"Acc. (FSNL) & {noveltyAcc[0][0]:.3f} \\footnotesize({noveltyAcc[0][1]:.3f}) & {noveltyAcc[1][0]:.3f} \\footnotesize({noveltyAcc[1][1]:.3f}) & "
        line +=             f"{noveltyAcc[2][0]:.3f} \\footnotesize({noveltyAcc[2][1]:.3f}) & {noveltyAcc[3][0]:.3f} \\footnotesize({noveltyAcc[3][1]:.3f}) & {noveltyAcc[4][0]:.3f} \\footnotesize({noveltyAcc[4][1]:.3f}) \\\\ \n"
        line += f"Precision   & {precision[0][0]:.3f} \\footnotesize({precision[0][1]:.3f}) & {precision[1][0]:.3f} \\footnotesize({precision[1][1]:.3f}) & "
        line +=             f"{precision[2][0]:.3f} \\footnotesize({precision[2][1]:.3f}) & {precision[3][0]:.3f} \\footnotesize({precision[3][1]:.3f}) & {precision[4][0]:.3f} \\footnotesize({precision[4][1]:.3f}) \\\\ \n"
        line += f"Recall      & {recall[0][0]:.3f} \\footnotesize({recall[0][1]:.3f}) & {recall[1][0]:.3f} \\footnotesize({recall[1][1]:.3f}) & "
        line +=             f"{recall[2][0]:.3f} \\footnotesize({recall[2][1]:.3f}) & {recall[3][0]:.3f} \\footnotesize({recall[3][1]:.3f}) & {recall[4][0]:.3f} \\footnotesize({recall[4][1]:.3f}) \\\\ \n"
        line += f"F1-score    & {F1[0][0]:.3f} \\footnotesize({F1[0][1]:.3f}) & {F1[1][0]:.3f} \\footnotesize({F1[1][1]:.3f}) & "
        line +=             f"{F1[2][0]:.3f} \\footnotesize({F1[2][1]:.3f}) & {F1[3][0]:.3f} \\footnotesize({F1[3][1]:.3f}) & {F1[4][0]:.3f} \\footnotesize({F1[4][1]:.3f}) \\\\ \n"
    
    print(title)
    print(line)

def dataForTableAvg2(data_df_novelty, fewShotClassifier, title, ways=[5, 10, 20, 30], n_shot=5):
    
    #print("dataForTableAvg", fewShotClassifier, ways, n_shot)
    # Header learn
    # ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween
    data_df_learn = data_df_novelty.loc[data_df_novelty['Novelty'] == False]
    data_df_learn = data_df_learn.loc[data_df_learn['FewShotClassifier'] == fewShotClassifier]
    data_df_learn = data_df_learn.loc[data_df_learn['Shot'] == n_shot]
    data_df_learn = data_df_learn.sort_values(by=['Way'])
    #print(data_df_learn['Accuracy'].to_list())
    #print("FSL K-way", data_df_learn['Way'].to_list())

    # Header novelty
    # ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,Seed,ModelName
    data_df = data_df_novelty.loc[data_df_novelty['Novelty'] == True]
    data_df = data_df.loc[data_df['FewShotClassifier'] == fewShotClassifier]
    data_df = data_df.loc[data_df['Shot'] == n_shot]
    data_df = data_df.sort_values(by=['Way'])
    #print(data_df['Accuracy'].to_list())
    #print("FSNL K-way", data_df['Way'].to_list())
    
    fewShotAcc = []
    noveltyAcc = []
    precision = []
    recall = []
    F1 = []
    #for n_way in [5, 10, 20, 30]:
    #for n_way in [5, 10, 20, 30, 40]:
    for n_way in ways:
        accuracyFSL = data_df_learn.loc[data_df_learn['Way'] == n_way]["Accuracy"].to_list()
        fewShotAcc.append([np.mean(accuracyFSL), np.std(accuracyFSL)*1000])
        data_df_way = data_df.loc[data_df['Way'] == n_way+1] # Novelty 6-way = 5-way + 1-novel
        accuracyFSNL = data_df_way["Accuracy"].to_list()
        noveltyAcc.append([np.mean(accuracyFSNL), np.std(accuracyFSNL)*1000])
        precisionFSNL = data_df_way["Precision"].to_list()
        precision.append([np.mean(precisionFSNL), np.std(precisionFSNL)*1000])
        recallFSNL = data_df_way["Recall"].to_list()
        recall.append([np.mean(recallFSNL), np.std(recallFSNL)*1000])
        F1FSNL = data_df_way["F1"].to_list()
        F1.append([np.mean(F1FSNL), np.std(F1FSNL)*1000])
    
    if len(ways) == 4:
        line = ""
        line += f"Acc. (FSL)  & {fewShotAcc[0][0]:.3f} ({fewShotAcc[0][1]:.1f}) & {fewShotAcc[1][0]:.3f} ({fewShotAcc[1][1]:.1f}) & "
        line +=             f"{fewShotAcc[2][0]:.3f} ({fewShotAcc[2][1]:.1f}) & {fewShotAcc[3][0]:.3f} ({fewShotAcc[3][1]:.1f}) \\\\ \n"
        line += f"Acc. (FSNL) & {noveltyAcc[0][0]:.3f} ({noveltyAcc[0][1]:.1f}) & {noveltyAcc[1][0]:.3f} ({noveltyAcc[1][1]:.1f}) & "
        line +=             f"{noveltyAcc[2][0]:.3f} ({noveltyAcc[2][1]:.1f}) & {noveltyAcc[3][0]:.3f} ({noveltyAcc[3][1]:.1f}) \\\\ \n"
        line += f"Precision   & {precision[0][0]:.3f} ({precision[0][1]:.1f}) & {precision[1][0]:.3f} ({precision[1][1]:.1f}) & "
        line +=             f"{precision[2][0]:.3f} ({precision[2][1]:.1f}) & {precision[3][0]:.3f} ({precision[3][1]:.1f}) \\\\ \n"
        line += f"Recall      & {recall[0][0]:.3f} ({recall[0][1]:.1f}) & {recall[1][0]:.3f} ({recall[1][1]:.1f}) & "
        line +=             f"{recall[2][0]:.3f} ({recall[2][1]:.1f}) & {recall[3][0]:.3f} ({recall[3][1]:.1f}) \\\\ \n"
        line += f"F1-score    & {F1[0][0]:.3f} ({F1[0][1]:.1f}) & {F1[1][0]:.3f} ({F1[1][1]:.1f}) & "
        line +=             f"{F1[2][0]:.3f} ({F1[2][1]:.1f}) & {F1[3][0]:.3f} ({F1[3][1]:.1f}) \\\\ \n"
    else:    
        line = ""
        line += f"Acc. (FSL)  & {fewShotAcc[0][0]:.3f} ({fewShotAcc[0][1]:.1f}) & {fewShotAcc[1][0]:.3f} ({fewShotAcc[1][1]:.1f}) & "
        line +=             f"{fewShotAcc[2][0]:.3f} ({fewShotAcc[2][1]:.1f}) & {fewShotAcc[3][0]:.3f} ({fewShotAcc[3][1]:.1f}) & {fewShotAcc[4][0]:.3f} ({fewShotAcc[4][1]:.1f}) \\\\ \n"
        line += f"Acc. (FSNL) & {noveltyAcc[0][0]:.3f} ({noveltyAcc[0][1]:.1f}) & {noveltyAcc[1][0]:.3f} ({noveltyAcc[1][1]:.1f}) & "
        line +=             f"{noveltyAcc[2][0]:.3f} ({noveltyAcc[2][1]:.1f}) & {noveltyAcc[3][0]:.3f} ({noveltyAcc[3][1]:.1f}) & {noveltyAcc[4][0]:.3f} ({noveltyAcc[4][1]:.1f}) \\\\ \n"
        line += f"Precision   & {precision[0][0]:.3f} ({precision[0][1]:.1f}) & {precision[1][0]:.3f} ({precision[1][1]:.1f}) & "
        line +=             f"{precision[2][0]:.3f} ({precision[2][1]:.1f}) & {precision[3][0]:.3f} ({precision[3][1]:.1f}) & {precision[4][0]:.3f} ({precision[4][1]:.1f}) \\\\ \n"
        line += f"Recall      & {recall[0][0]:.3f} ({recall[0][1]:.1f}) & {recall[1][0]:.3f} ({recall[1][1]:.1f}) & "
        line +=             f"{recall[2][0]:.3f} ({recall[2][1]:.1f}) & {recall[3][0]:.3f} ({recall[3][1]:.1f}) & {recall[4][0]:.3f} ({recall[4][1]:.1f}) \\\\ \n"
        line += f"F1-score    & {F1[0][0]:.3f} ({F1[0][1]:.1f}) & {F1[1][0]:.3f} ({F1[1][1]:.1f}) & "
        line +=             f"{F1[2][0]:.3f} ({F1[2][1]:.1f}) & {F1[3][0]:.3f} ({F1[3][1]:.1f}) & {F1[4][0]:.3f} ({F1[4][1]:.1f}) \\\\ \n"
    
    print(title)
    print(line)
    
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
        

def dataPaperV2():

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
    dataForTable(data_df_10_novelty, data_df_10_learn, fewShotClassifier, "EU Moths (Alpha=1.0)", ways=[5, 10, 20, 30, 38])

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
        
def dataPaperFinal():

    fewShotClassifier = "Prototypical"
    
    data_df_novelty = pd.read_csv("./paperData/resnet12_Omniglot_0_novelty_rand_test.txt")
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "Omniglot (Alpha=0.0, N-shot=5)", ways=[5, 10, 20, 30], n_shot=5)   
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "Omniglot (Alpha=0.0, N-shot=1)", ways=[5, 10, 20, 30], n_shot=1) 

    data_df_novelty = pd.read_csv("./paperData/resnet12_Omniglot_10_novelty_rand_test.txt")
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "Omniglot (Alpha=1.0, N-shot=5)", ways=[5, 10, 20, 30], n_shot=5)   
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "Omniglot (Alpha=1.0, N-shot=1)", ways=[5, 10, 20, 30], n_shot=1) 
    
    data_df_novelty = pd.read_csv("./paperData/resnet18_miniImagenet_1_novelty_rand_test.txt")
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "MiniImageNet (Alpha=0.1, N-shot=5)", ways=[5, 10, 15, 19], n_shot=5)   
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "MiniImageNet (Alpha=0.1, N-shot=1)", ways=[5, 10, 15, 19], n_shot=1)       

    data_df_novelty = pd.read_csv("./paperData/resnet18_euMoths_10_novelty_rand_test.txt")
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "EU Moths (Alpha=1.0, N-shot=5)", ways=[5, 10, 20, 30, 40], n_shot=5)   
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "EU Moths (Alpha=1.0, N-shot=1)", ways=[5, 10, 20, 30, 40], n_shot=1)   
    
    data_df_novelty = pd.read_csv("./paperData/resnet18_CUB_10_novelty_rand_test.txt")
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "CUB (Alpha=1.0, N-shot=5)", ways=[5, 10, 20, 29], n_shot=5)   
    dataForTableAvg1(data_df_novelty, fewShotClassifier, "CUB (Alpha=1.0, N-shot=1)", ways=[5, 10, 20, 29], n_shot=1)       

    
#%% MAIN
if __name__=='__main__':
    

    #dataPaperV2()
    dataPaperFinal()