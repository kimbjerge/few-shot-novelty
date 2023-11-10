# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 09:31:32 2023

@author: Kim Bjerge
"""

import os
import random
import numpy as np
import torch
import argparse
#from torch import nn
import pandas as pd
from torch.utils.data import DataLoader

#from easyfsl.datasets import CUB
#from easyfsl.datasets import MiniImageNet
#from easyfsl.datasets import EasySet
#from easyfsl.modules import resnet12
from PrototypicalNetworksNovelty import PrototypicalNetworksNovelty
from utilsNovelty import evaluate, Metrics
from NoveltyThreshold import getLearnedThreshold, StdTimesTwoThredshold

from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks, RelationNetworks, MatchingNetworks, TransductiveFinetuning
from easyfsl.methods import SimpleShot, Finetune, FEAT, BDCSPN, LaplacianShot, PTMAP, TIM
from easyfsl.samplers import TaskSampler

from FewShotModelData import EmbeddingsModel, FewShotDataset

from torchvision.models import resnet50 #, ResNet50_Weights
from torchvision.models import resnet34 #, ResNet34_Weights
from torchvision.models import resnet18 #, ResNet18_Weights


def load_model(argsModel, argsWeights, argsAlpha):
    
    if argsModel == 'resnet50':
        #print('resnet50')
        #ResNetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        ResNetModel = resnet50(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        #modelName = "./models/Resnet50_"+argsWeights+"_model.pth"
        #modelName = "./models/Resnet50_"+argsWeights+"_classic_pretrained.pth"
        
        #modelName = "./modelsAdv/Resnet50_"+argsWeights+"_episodic_AdvLoss_E61_A0_824.pth"
        #modelName = "./modelsAdv/Resnet50_"+argsWeights+"_episodic_AdvLoss_E367_A0_836.pth"
        modelName = "./modelsAdv/Resnet50_"+argsWeights+"_episodic_AdvLoss_E1029_A0_922.pth"
        feat_dim = 2048
    if argsModel == 'resnet34':
        #print('resnet34')
        ResNetModel = resnet34(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        #modelName = "./models/Resnet34_"+argsWeights+"_model.pth"
        #modelName = "./models/Resnet34_"+argsWeights+"_classic_pretrained.pth"
        
        modelName = "./modelsAdv/Resnet34_"+argsWeights+"_episodic_AdvLoss_E1223_A0_877.pth"
        feat_dim = 512
    if argsModel == 'resnet18':
        #print('resnet18')
        #ResNetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        ResNetModel = resnet18(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        #modelName = "./models/Resnet18_"+argsWeights+"_model.pth"
        #modelName = "./models/Resnet18_"+argsWeights+"_episodicV1_0_7520.pth" # Weights saved in few_shot format
        #modelName = "./models/Resnet18_"+argsWeights+"_episodicV1_0_8644_1318.pth" # Weights saved in few_shot format
        #modelName = "./models/Resnet18_"+argsWeights+"_classic_pretrained.pth"
        modelName = "./models/Resnet18_"+argsWeights+"_episodic_0_7105_5w.pth" # miniImagenet
        #modelName = "./models/Resnet18_"+argsWeights+"_epi_model.pth"
        feat_dim = 512
    if argsModel == 'resnet12':
        #print('resnet12')
        model = resnet12(use_fc=True, num_classes=num_classes) #.to(DEVICE)
        #modelName = "./models/Resnet12_"+args.weights+"_model.pth" #"_model.pth" Ecuclidian distance used during training
        #modelName = "./modelsAdv200/Resnet12_"+args.weights+"_episodic_ScatterEuclidianLoss.pth" # 124 epochs, 96.93 % accuracy
        #modelName = "./modelsAdv200/Resnet12_"+args.weights+"_episodic_EntropyEuclidianLoss.pth" # ~100 epochs, 92.87 % accuracy
        #modelName = "./modelsAdv200/Resnet12_"+args.weights+"_episodic_MeanLoss.pth" #
        #modelName = "./modelsAdv200/Resnet12_"+args.weights+"_episodic_VarLoss.pth" #
        #modelName = "./modelsAdv200/Resnet12_"+args.weights+"_episodic_ScatterLoss.pth" # best model after 136 epochs 85.5% accuracy

        #modelName = "./modelsAdv200/Resnet12_"+args.weights+"_episodic_2_AdvLoss.pth" # 145 epochs, 92.87 % accuracy
        #modelName = "./modelsAdv200/Resnet12_"+args.weights+"_episodic_0_AdvLoss.pth" # 145 epochs, 92.87 % accuracy
        #modelName = "./modelsAdv/Resnet12_"+args.weights+"_episodic_0_AdvLoss.pth" # 1436 epochs, 98.75 % accuracy
        #modelName = "./modelsAdv/Resnet12_"+args.weights+"_episodic_1_AdvLoss.pth" # 1261 epochs, 98.85 % accuracy
        #modelName = "./modelsAdv/Resnet12_"+args.weights+"_episodic_2_AdvLoss.pth" # 1335 epochs, 99.07 % accuracy
        modelName = "./modelsAdv/Resnet12_"+args.weights+"_episodic_"+ str(int(argsAlpha*10)) + "_AdvLoss.pth" 
        feat_dim = 64
    
    #modelName = "./models/Resnet18_euMoths_state.pth"
    #model.load_state_dict(torch.load(modelName))
    
    if "model.pth" in modelName or "_classic_" in modelName or "_episodic_" in modelName:
        if args.weights == 'ImageNet':
            print('Using pretrained weights with ImageNet dataset')
        else:
            print('Using saved model weights', modelName)
            modelSaved = torch.load(modelName, map_location=torch.device(DEVICE))
            #ResNetModel.load_state_dict(modelSaved.state_dict())
            model.load_state_dict(modelSaved.state_dict())

    if "episodicV1" in modelName:
        if args.weights == 'ImageNet':
            print('Using pretrained weights with ImageNet dataset')
        else:
            # Weights for entier few shot model is saved - load and convert weights
            print('Using saved model from episodic saved weights', modelName)
            modelSaved = torch.load(modelName, map_location=torch.device(DEVICE))
            #ResNetModel.load_state_dict(modelSaved.state_dict())
            few_shot_classifier = PrototypicalNetworks(model)
            few_shot_classifier.load_state_dict(modelSaved.state_dict())

    model.eval()
    model = model.to(DEVICE)
    
    return model, modelName, feat_dim


def load_test_dataset(argsDataset, argsLearning):
    
    if args.dataset == 'Omniglot':
        if args.learning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirOmniglot, training=False)
            print("Omniglot Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirOmniglot, training=False)
            print("Omniglot Test dataset")
    if args.dataset == 'euMoths':
        #test_set = FewShotDataset(split="train", image_size=image_size, root=dataDirEuMoths,training=False)
        if args.learning:
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirEuMoths, training=False)
            print("euMoths Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirEuMoths, training=False)
            print("euMoths Test dataset")
    if args.dataset == 'CUB':
        #test_set = FewShotDataset(split="train", image_size=image_size, root=dataDirCUB, training=False)
        if args.learning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirCUB, training=False)
            print("CUB Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirCUB, training=False)
            print("CUB Test dataset")
    if args.dataset == 'miniImagenet':
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', specs_file=dataDirMiniImageNet+'/test.csv', image_size=image_size, training=False)
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', split="test", image_size=image_size, training=False)
        if args.learning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirMiniImageNet, training=False)
            print("miniImageNet Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirMiniImageNet, training=False)
            print("miniImageNet Test dataset")
    
    return test_set

def get_threshold_learned(modelName, argsModel, argsWeights, nameNoveltyLearned, n_shot, useBayesThreshold=True):
    
    noveltyLearnedFile = "./learnedAdv/" + argsModel + '_' + argsWeights + nameNoveltyLearned + '.csv'
    print("Learned threshold file", noveltyLearnedFile)
    
    df = pd.read_csv(noveltyLearnedFile)
    
    df = df.loc[df['ModelName'] == modelName]
    df = df.loc[df['Shot'] == n_shot]
    
    if useBayesThreshold:
        threshold = df['BayesThreshold'].to_numpy()[0]
        print("Bayes threshold", threshold)
    else:
        threshold = StdTimesTwoThredshold(df['Average'].to_numpy()[0], df['Std'].to_numpy()[0])
        print("Std threshold", threshold)
        
    return threshold

def test_or_learn(test_set, test_sampler, few_shot_classifier, 
                  novelty_th, use_novelty, learn_th, n_workers, metric, DEVICE):

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    
    
    accuracy, learned_th, avg, std, avg_o, std_o = evaluate(few_shot_classifier, 
                                                            test_loader, 
                                                            novelty_th, 
                                                            device=DEVICE,
                                                            tqdm_prefix="Test",
                                                            plt_hist=True,
                                                            use_novelty=use_novelty, 
                                                            metric=metric,
                                                            learn_th=learn_th, 
                                                            )
    
    print(f"Average accuracy : {(100 * accuracy):.2f} %")
    return accuracy, learned_th, avg, std, avg_o, std_o


#%% MAIN
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='resnet12') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    parser.add_argument('--weights', default='Omniglot') #ImageNet, mini_imagenet, euMoths, CUB, Omniglot
    parser.add_argument('--dataset', default='Omniglot') #miniImagenet, euMoths, CUB, Omniglot
    
    # parser.add_argument('--model', default='resnet34') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    # parser.add_argument('--weights', default='euMoths') #ImageNet, mini_imagenet, euMoths, CUB, Omniglot
    # parser.add_argument('--dataset', default='euMoths') #miniImagenet, euMoths, CUB, Omniglot
    
    # parser.add_argument('--model', default='resnet18') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    # parser.add_argument('--weights', default='mini_imagenet') #ImageNet, mini_imagenet, euMoths, CUB, Omniglot
    # parser.add_argument('--dataset', default='miniImagenet') #miniImagenet, euMoths, CUB, Omniglot
    
    # parser.add_argument('--model', default='resnet12') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    # parser.add_argument('--weights', default='Omniglot') #ImageNet, euMoths, CUB, Omniglot
    # parser.add_argument('--dataset', default='Omniglot') #miniImagenet, euMoths, CUB, Omniglot
    
    parser.add_argument('--novelty', default='', type=bool) #default false when no parameter - automatic False when learning True
    parser.add_argument('--learning', default='', type=bool) #default false when no parameter - learn threshold for novelty detection
    parser.add_argument('--shot', default=5, type=int) 
    parser.add_argument('--way', default=5, type=int) # Way 0 is novelty class
    parser.add_argument('--query', default=6, type=int)
    parser.add_argument('--alpha', default=0.7, type=float)
    parser.add_argument('--threshold', default='bayes') # bayes or std threshold to be used
    args = parser.parse_args()
  
    resDir = "./result/"
    dataDirMiniImageNet = "./data/mini_imagenet"
    dataDirEuMoths = "./data/euMoths"
    dataDirCUB = "./data/CUB"
    dataDirOmniglot = "./data/Omniglot"
    subDir = "testAdv/"

    if args.learning:
        args.novelty = False # Novelty detection is disabled during learning threshold
        args.way = 5
        
    if args.novelty:
        args.way += 1 # Added one additional way used for outlier class 
        
    print(args.model, args.weights, args.dataset, args.novelty, args.learning, args.shot, args.way, args.query, args.alpha, args.threshold)
    #image_size = 28 # Omniglot
    #image_size = 84 # CUB dataset

    if args.model == 'resnet12':
        image_size = 28 # Omniglot dataset
    else:
        image_size = 224 # ResNet euMoths

    #image_size = 300 # EfficientNet B3
    #image_size = 380 # EfficientNet B4
    #image_size = 600 # EfficientNet B7

    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    n_workers = 12
    
    n_way = args.way
    n_shot = args.shot
    n_query = args.query

    num_classes = 100  
    if args.weights == 'CUB':
        num_classes = 140  
    if args.weights == 'Omniglot':
        num_classes = 3856  
    if args.weights == 'mini_imagenet':
        num_classes = 60
        
    if args.learning:
        n_test_tasks = 50 # 50 learning on validation
    else:
        n_test_tasks = 100 # 500 test
        
   
    #%% Create model and prepare for training
    #DEVICE = "cuda"
    DEVICE = "cpu"
    
    # model = resnet12(
    #     use_fc=True,
    #     num_classes=num_classes,
    # ).to(DEVICE)
    
    model, modelName, feat_dim = load_model(args.model, args.weights, args.alpha)

    #few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)
    
    if args.learning:
        few_shot_classifiers =  [ 
                                 ["PrototypicalNetworksNovelty", PrototypicalNetworksNovelty(model, use_normcorr=1)]
                                # ["PrototypicalNetworksNovelty", PrototypicalNetworksNovelty(model, use_normcorr=3)]
                                ]
    else:
        few_shot_classifiers =  [ 
                                 # #["RelationNetworks", RelationNetworks(model, feature_dimension=3)], No
                                 ["Prototypical", PrototypicalNetworksNovelty(model, use_normcorr=1)],
                                 # ["PrototypicalNetworksNovelty", PrototypicalNetworksNovelty(model, use_normcorr=3)],
                                 #["PrototypicalNetworks", PrototypicalNetworks(model)], #No
                                 #["MatchingNetworks", MatchingNetworks(model, feature_dimension=feat_dim)], No - special
                                 #["TransductiveFinetuning", TransductiveFinetuning(model)],  No - l2
                                 #["SimpleShot", SimpleShot(model)], No - too simple
                                 ["Finetune", Finetune(model)], 
                                 # #["FEAT", FEAT(model)], - error few-shot and novelty
                                 ["BD-CSPN", BDCSPN(model)], 
                                 #["LaplacianShot", LaplacianShot(model)], No - special
                                 # #["PT-MAP", PTMAP(model)], No
                                 ["TIM", TIM(model)]
                                ]
    
    test_set = load_test_dataset(args.dataset, args.learning)
    
    #subDir = args.weights + '/'
    if os.path.exists(resDir+subDir) == False:
        os.mkdir(resDir+subDir)
        print("Create result directory", resDir+subDir)
        
    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
    )
    
    #test(model, test_set, test_sampler, few_shot_classifier, n_workers)
    novelty_th = getLearnedThreshold(args.weights, args.model, args.shot)    
            
    if args.learning:     
        resFileName = args.model + '_' + args.dataset + "_novelty_learn.csv"
        line = "ModelName,FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier\n"
    else:
        resFileName =  args.model + '_' +  args.dataset + "_novelty_test.txt"
        line = "Model,FewShotClassifier,Way,Shot,Query,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName\n"
        if "bayes" in args.threshold:
            useBayesThreshold = True
            args.threshold = "bayes"
        else:
            useBayesThreshold = False
            args.threshold = "std"
        if args.model == 'resnet12': # More models to be learned
            novelty_th = get_threshold_learned(modelName, args.model, args.weights, "_episodic_novelty_learned", args.shot, useBayesThreshold=useBayesThreshold)
        else:
            useBayesThreshold = False
            args.threshold = "std"
            

    if os.path.exists(resDir+subDir+resFileName):
        resFile = open(resDir+subDir+resFileName, "a")
    else:
        resFile = open(resDir+subDir+resFileName, "w")
        print(line)
        resFile.write(line)
        resFile.flush()                 
    
    #novelty_th = 0.8739 # Omniglot, 5-way 5-shot, adv, alpha 0.2, _episodic_2_AdvLoss.pth (145 Epochs)
    #novelty_th = 0.8454 # Omniglot, 5-way 5-shot, adv, alpha 0, _episodic_0_AdvLoss.pth (145 Epochs)

    #novelty_th = -11.90  # Omniglot, 5-way 5-shot, adv, alpha 0, _episodic_0_AdvLoss.pth (1436 Epochs) - Euclidian - accuracy drops to 0.825 (use_normcorr=3)
    #novelty_th = 0.8405  # Omniglot, 5-way 5-shot, adv, alpha 0, _episodic_0_AdvLoss.pth (1436 Epochs)
    #novelty_th = 0.8534  # Omniglot, 5-way 5-shot, adv, alpha 0.1, _episodic_1_AdvLoss.pth (1436 Epochs)
    #novelty_th = 0.8532  # Omniglot, 5-way 5-shot, adv, alpha 0.2, _episodic_2_AdvLoss.pth (1436 Epochs)
    
    #novelty_th = 0.8392  # euMoths, 5-way 5-shot, ImageNet, accuracy = 0.929, novelty = 0.775,  (resnet50)
    #novelty_th = 0.8244  # euMoths, 5-way 5-shot, adv, alpha 0.2, _episodic_AdvLoss_E61_A0_824.pth, accuracy = 0.8607, novelty = 0.736,  (resnet50)
    #novelty_th = 0.8314  # euMoths, 5-way 5-shot, adv, alpha 0.2, _episodic_AdvLoss_E1029_A0_922.pth (resnet50)
    #novelty_th = 0.7494  # euMoths, 5-way 5-shot, adv, alpha 0.2, _episodic_AdvLoss_E1223_A0_877.pth (resnet34) few-shot 0.897 and novelty 0.801 
    novelty_th = 0.8148   # euMoths, 5-way 5-shot, ImageNet, accuracy = 0.945, novelty = 0.775, 
    
    if args.learning:     
    
        for few_shot in few_shot_classifiers:
            print(few_shot[0])
            print("Use softmax", few_shot[1].use_softmax)
            few_shot_classifier = few_shot[1].to(DEVICE)
            accuracy, threshold, avg, std, avg_o, std_o  = test_or_learn(test_set, test_sampler, few_shot_classifier, 
                                                                         novelty_th, args.novelty, args.learning, 
                                                                         n_workers, None, DEVICE)
            line = modelName + ',' + few_shot[0] + ',' + str(args.way) + ','  + str(args.shot) + ','  + str(args.query) + ',' + str(accuracy) + ',' 
            line += str(threshold) + ',' + str(avg) + ',' + str(std) + ',' + str(avg_o) + ',' + str(std_o) + '\n'
            print(line)
            resFile.write(line)    
            resFile.flush()                 
    
        resFile.close()
        print("Result saved to", resFileName)
        
    else: # Testing
                      
        for few_shot in few_shot_classifiers:
            print(few_shot[0])
            print("Use softmax", few_shot[1].use_softmax)
            few_shot_classifier = few_shot[1].to(DEVICE)
            metric = Metrics()
            accuracy, threshold, avg, std, avg_o, std_o = test_or_learn(test_set, test_sampler, few_shot_classifier, 
                                                                        novelty_th, args.novelty, args.learning, 
                                                                        n_workers, metric, DEVICE)
            line = args.model + ',' + few_shot[0] + ',' + str(n_way) + ','  + str(n_shot) + ','  + str(n_query) + ',' 
            line += str(accuracy) + ',' + str(metric.precision())  + ',' + str(metric.recall()) + ',' + str(metric.f1score()) + ','
            line += str(metric.TP()) + ',' + str(metric.FP()) + ',' + str(metric.FN()) + ','
            line += args.threshold + ',' + str(threshold) + ',' + str(args.alpha) + ',' + modelName +  '\n'
            print(line)
            resFile.write(line)    
            resFile.flush()
            
        resFile.close()
        print("Result saved to", resFileName)
        
    
        

