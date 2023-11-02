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
from torch.utils.data import DataLoader

#from easyfsl.datasets import CUB
#from easyfsl.datasets import MiniImageNet
#from easyfsl.datasets import EasySet
#from easyfsl.modules import resnet12
from PrototypicalNetworksNovelty import PrototypicalNetworksNovelty
from utilsNovelty import evaluate
from NoveltyThreshold import getLearnedThreshold, BayesTwoClassThreshold, StdTimesTwoThredshold

from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks, RelationNetworks, MatchingNetworks, TransductiveFinetuning
from easyfsl.methods import SimpleShot, Finetune, FEAT, BDCSPN, LaplacianShot, PTMAP, TIM
from easyfsl.samplers import TaskSampler

from FewShotModelData import EmbeddingsModel, FewShotDataset

from torchvision.models import resnet50 #, ResNet50_Weights
from torchvision.models import resnet34 #, ResNet34_Weights
from torchvision.models import resnet18 #, ResNet18_Weights


def load_model(argsModel, argsWeights):

    if argsModel == 'resnet50':
        print('resnet50')
        #ResNetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        ResNetModel = resnet50(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        modelName = "./models/Resnet50_"+argsWeights+"_model.pth"
        feat_dim = 2048
    if argsModel == 'resnet34':
        print('resnet34')
        ResNetModel = resnet34(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        modelName = "./models/Resnet34_"+argsWeights+"_model.pth"
        feat_dim = 512
    if argsModel == 'resnet18':
        print('resnet18')
        #ResNetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        ResNetModel = resnet18(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        modelName = "./models/Resnet18_"+argsWeights+"_model.pth"
        feat_dim = 512
    if argsModel == 'resnet12':
        print('resnet12')
        model = resnet12(use_fc=True, num_classes=num_classes) #.to(DEVICE)
        modelName = "./models/Resnet12_"+argsWeights+"_model.pth" #"_model.pth"
        feat_dim = 64
    
    if argsWeights == 'ImageNet':
        print('Using pretrained weights with ImageNet dataset')
    else:
        print('Using saved model weights', modelName)
        modelSaved = torch.load(modelName, map_location=torch.device(DEVICE))
        #ResNetModel.load_state_dict(modelSaved.state_dict())
        model.load_state_dict(modelSaved.state_dict())
 
    subDir = args.weights + '/'
    if os.path.exists(resDir+subDir) == False:
        os.mkdir(resDir+subDir)
        print("Create result directory", resDir+subDir)

    model.eval()
    model = model.to(DEVICE)
    
    return model, modelName, feat_dim


def load_test_dataset(argsDataset, argsLearning):

    if argsDataset == 'Omniglot':
        if argsLearning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirOmniglot, training=False)
            print("Omniglot Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirOmniglot, training=False)
            print("Omniglot Test dataset")
    if argsDataset == 'euMoths':
        #test_set = FewShotDataset(split="train", image_size=image_size, root=dataDirEuMoths,training=False)
        if argsLearning:
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirEuMoths, training=False)
            print("euMoths Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirEuMoths, training=False)
            print("euMoths Test dataset")
    if argsDataset == 'CUB':
        #test_set = FewShotDataset(split="train", image_size=image_size, root=dataDirCUB, training=False)
        if argsLearning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirCUB, training=False)
            print("CUB Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirCUB, training=False)
            print("CUB Test dataset")
    if argsDataset == 'miniImagenet':
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', specs_file=dataDirMiniImageNet+'/test.csv', image_size=image_size, training=False)
        #test_set = MiniImageNet(root=dataDirMiniImageNet+'/images', split="test", image_size=image_size, training=False)
        if argsLearning:       
            test_set = FewShotDataset(split="val", image_size=image_size, root=dataDirMiniImageNet, training=False)
            print("miniImageNet Val dataset")
        else:
            test_set = FewShotDataset(split="test", image_size=image_size, root=dataDirMiniImageNet, training=False)
            print("miniImageNet Val dataset")
            
    return test_set


def test_or_learn(test_set, test_sampler, few_shot_classifier, 
                  novelty_th, use_novelty, learn_th, n_workers, DEVICE):

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    
    accuracy, learned_th, avg, std, avg_o, std_o = evaluate(few_shot_classifier, test_loader, 
                                                            novelty_th, 
                                                            use_novelty=use_novelty, 
                                                            learn_th=learn_th, 
                                                            device=DEVICE, tqdm_prefix="Test")
    print(f"Average accuracy : {(100 * accuracy):.2f} %")
    return accuracy, learned_th, avg, std, avg_o, std_o


#%% MAIN
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default='resnet50') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    # parser.add_argument('--weights', default='euMoths') #ImageNet, euMoths, CUB, Omniglot
    # parser.add_argument('--dataset', default='euMoths') #miniImagenet, euMoths, CUB, Omniglot
    parser.add_argument('--model', default='resnet12') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    parser.add_argument('--weights', default='Omniglot') #ImageNet, euMoths, CUB, Omniglot
    parser.add_argument('--dataset', default='Omniglot') #miniImagenet, euMoths, CUB, Omniglot
    parser.add_argument('--novelty', default='True', type=bool) #default false when no parameter - automatic False when learning True
    parser.add_argument('--learning', default='True', type=bool) #default false when no parameter - learn threshold for novelty detection
    parser.add_argument('--shot', default=5, type=int) 
    parser.add_argument('--way', default=5, type=int) # Way 0 is novelty class
    parser.add_argument('--query', default=6, type=int)
    args = parser.parse_args()
  
    resDir = "./result/"
    dataDirMiniImageNet = "./data/mini_imagenet"
    dataDirEuMoths = "./data/euMoths"
    dataDirCUB = "./data/CUB"
    dataDirOmniglot = "./data/Omniglot"
    subDir = ""

    if args.learning:
        args.novelty = False # Novelty detection is disabled during learning threshold
        
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
        
    if args.learning:
        n_test_tasks = 50 # 50 learning on validation
    else:
        n_test_tasks = 500 # 500 test
        
   
    #%% Create model and prepare for training
    #DEVICE = "cuda"
    DEVICE = "cpu"
    
    # model = resnet12(
    #     use_fc=True,
    #     num_classes=num_classes,
    # ).to(DEVICE)
    
    model, modelName, feat_dim = load_model(args.model, args.weights)

    #few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)
    
    if args.learning:
        few_shot_classifiers =  [ 
                                 ["PrototypicalNetworksNovelty", PrototypicalNetworksNovelty(model, use_normcorr=1)]
                                ]
    else:
        few_shot_classifiers =  [ 
                                 # #["RelationNetworks", RelationNetworks(model, feature_dimension=3)], No
                                 ["PrototypicalNetworksNovelty", PrototypicalNetworksNovelty(model, use_normcorr=1)],
                                 #["PrototypicalNetworks", PrototypicalNetworks(model)], No
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
    
    
    #test(model, test_set, test_sampler, few_shot_classifier, n_workers)
    if args.learning:     
        resFileName = args.dataset + '_' + args.model + "_novelty_learned.txt"
        resFile = open(resDir+subDir+resFileName, "a")
        
        few_shot = few_shot_classifiers[0]
        print(few_shot[0])
        few_shot_classifier = few_shot[1].to(DEVICE)
        novelty_th = 0.5

        line = "FewShotClassifier,Way,Shot,Query,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier\n"
        print(line)
        resFile.write(line)   
        
        for n_shot in [1,2,3,4,5,6,7,8,9]: # Learn distribution for each shot in range 1-9
        
            test_sampler = TaskSampler(
                test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
            )

            accuracy, threshold, avg, std, avg_o, std_o = test_or_learn(test_set, test_sampler, few_shot_classifier, 
                                                                        novelty_th, args.novelty, args.learning, 
                                                                        n_workers, DEVICE)

            line = few_shot[0] + ',' + str(n_way) + ','  + str(n_shot) + ','  + str(n_query) + ',' + str(accuracy) + ',' + str(threshold) + ',' + str(avg) + ',' + str(std) + ',' + str(avg_o) + ',' + str(std_o) + '\n'
            print(line)
            resFile.write(line)    
            
        resFile.close()
        print("Result saved to", resFileName)
            
    else: #Testing
        resFileName = args.dataset + '_' + args.model + '_' + str(n_way) + 'way_' + str(n_shot) +"shot_novelty_test.txt"
        resFile = open(resDir+subDir+resFileName, "w")

        test_sampler = TaskSampler(
            test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
        )
        
        novelty_th = getLearnedThreshold(args.weights, args.model, args.shot)
    
        line = "FewShotClassifier,Way,Shot,Query,Accuracy,Threshold,Average,Std\n"
        resFile.write(line)   
        for few_shot in few_shot_classifiers:
            print(few_shot[0])
            print("Use softmax", few_shot[1].use_softmax)
            few_shot_classifier = few_shot[1].to(DEVICE)
            accuracy, threshold, avg, std, avg_o, std_o = test_or_learn(test_set, test_sampler, few_shot_classifier, 
                                                                        novelty_th, args.novelty, args.learning, 
                                                                        n_workers, DEVICE)
            line = few_shot[0] + ',' + str(n_way) + ','  + str(n_shot) + ','  + str(n_query) + ',' + str(accuracy) + ',' + str(threshold) + ',' + str(avg) + ',' + str(std) + '\n'
            resFile.write(line)    
        resFile.close()
        print("Result saved to", resFileName)
        

