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

from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks, RelationNetworks, MatchingNetworks, TransductiveFinetuning
from easyfsl.methods import SimpleShot, Finetune, FEAT, BDCSPN, LaplacianShot, PTMAP, TIM
from easyfsl.samplers import TaskSampler

from FewShotModelData import EmbeddingsModel, FewShotDataset

from torchvision.models import resnet50 #, ResNet50_Weights
from torchvision.models import resnet34 #, ResNet34_Weights
from torchvision.models import resnet18 #, ResNet18_Weights

"""
ImageNetNoveltyTh = {# CPU
        'resnet18': 0.6972,
        'resnet34': 0.6974,
        'resnet50': 0.6196  # 0.7287 5 shot
        }
ImageNetNoveltyThGPU = {# GPU other default weights than CPU???
        'resnet18': 0.8088,
        'resnet34': 0.8115,
        'resnet50': 0.8468
        }

euMothsNoveltyTh = {# GPU     CPU
        'resnet18': 0.7580,   #0.7582
        'resnet34': 0.7537,   #0.7593
        'resnet50': 0.7948,   #0.7932
        } 
CUBNoveltyTh = {    # CPU     GPU ????
        'resnet18': 0.7189,   #0.8128
        'resnet34': 0.7146,   #0.8194
        'resnet50': 0.7581    #0.8471
        } 
"""

# Learned avg and std on validation dataset with 1-shot
ImageNetNoveltyAvgStd = {# avg, std
         'resnet18': [0.72226, 0.07111], 
         'resnet34': [0.73021, 0.07541], 
         'resnet50': [0.76526, 0.07281]  # Th 0.6196 1 shot, TH 0.7287 5 shot
         } 
euMothsNoveltyAvgStd = {# avg, std
         'resnet18': [0.77879, 0.05930], 
         'resnet34': [0.77685, 0.06190], 
         'resnet50': [0.80555, 0.05177]  
         } 
CUBNoveltyAvgStd = {# avg, std
         'resnet18': [0.71100, 0.05190], 
         'resnet34': [0.71902, 0.05467], 
         'resnet50': [0.76377, 0.04995]  
         }
OmniglotNoveltyAvgStd = {# avg, std
         'resnet12': [0.71100, 0.05190],
         'resnet18': [0.71100, 0.05190], 
         'resnet34': [0.71902, 0.05467], 
         'resnet50': [0.76377, 0.04995]  
         }  

def test_or_learn(test_set, test_sampler, few_shot_classifier, 
                  novelty_th, use_novelty, learn_th, n_workers, DEVICE):

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    
    accuracy, learned_th = evaluate(few_shot_classifier, test_loader, 
                                    novelty_th, 
                                    use_novelty=use_novelty, 
                                    learn_th=learn_th, 
                                    device=DEVICE, tqdm_prefix="Test")
    print(f"Average accuracy : {(100 * accuracy):.2f} %")
    return accuracy, learned_th

def getLearnedThreshold(weightsName, modelName, n_shot):
    
    novelty_th = 0
    if weightsName == 'ImageNet':
        avg = ImageNetNoveltyAvgStd[modelName][0]
        std = ImageNetNoveltyAvgStd[modelName][1]
        #novelty_th = ImageNetNoveltyTh[modelName]
    if weightsName == 'Omniglot':
        avg = OmniglotNoveltyAvgStd[modelName][0]
        std = OmniglotNoveltyAvgStd[modelName][1]
    if weightsName == 'euMoths':
        avg = euMothsNoveltyAvgStd[modelName][0]
        std = euMothsNoveltyAvgStd[modelName][1]
        #novelty_th = euMothsNoveltyTh[modelName]
    if weightsName == 'CUB':
        avg = CUBNoveltyAvgStd[modelName][0]
        std = CUBNoveltyAvgStd[modelName][1]
        #novelty_th = CUBNoveltyTh[modelName]

    novelty_th = avg - 2*(std/np.sqrt(n_shot)) # Mean filter sigma/sqrt(M)
    print("Novelty threshold", weightsName, modelName, avg, std, novelty_th)
    return novelty_th

#%% MAIN
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default='resnet18') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    #parser.add_argument('--weights', default='ImageNet') #ImageNet, euMoths, CUB, Omniglot
    #parser.add_argument('--dataset', default='miniImagenet') #miniImagenet, euMoths, CUB, Omniglot
    parser.add_argument('--model', default='resnet12') #resnet12 (Omniglot), resnet18, resnet34, resnet50
    parser.add_argument('--weights', default='Omniglot') #ImageNet, euMoths, CUB, Omniglot
    parser.add_argument('--dataset', default='Omniglot') #miniImagenet, euMoths, CUB, Omniglot
    parser.add_argument('--novelty', default='True', type=bool) #default false when no parameter
    parser.add_argument('--learning', default='', type=bool) #default false when no parameter - learn threshold for novelty detection
    parser.add_argument('--shot', default=5, type=int)
    parser.add_argument('--way', default=6, type=int) # Way 0 is novelty class
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
    DEVICE = "cuda"
    #DEVICE = "cpu"
    
    # model = resnet12(
    #     use_fc=True,
    #     num_classes=num_classes,
    # ).to(DEVICE)
    
    if args.model == 'resnet50':
        print('resnet50')
        #ResNetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        ResNetModel = resnet50(pretrained=True) # 80.86, 25.6M
        modelName = "./models/Resnet50_"+args.weights+"_model.pth"
        feat_dim = 2048
    if args.model == 'resnet34':
        print('resnet34')
        ResNetModel = resnet34(pretrained=True) # 80.86, 25.6M
        modelName = "./models/Resnet34_"+args.weights+"_model.pth"
        feat_dim = 512
    if args.model == 'resnet18':
        print('resnet18')
        #ResNetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        ResNetModel = resnet18(pretrained=True) # 80.86, 25.6M
        modelName = "./models/Resnet18_"+args.weights+"_model.pth"
        feat_dim = 512
    if args.model == 'resnet12':
        print('resnet12')
        ResNetModel = resnet12(use_fc=True, num_classes=num_classes) #.to(DEVICE)
        modelName = "./models/Resnet12_"+args.weights+"_classic.pth" #"_model.pth"
        feat_dim = 512

    model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
    
    #modelName = "./models/Resnet18_euMoths_state.pth"
    #model.load_state_dict(torch.load(modelName))
    if args.weights == 'ImageNet':
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

    #few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)
 
    few_shot_classifiers =  [ 
                             # #["RelationNetworks", RelationNetworks(model, feature_dimension=3)], No
                             ["PrototypicalNetworksNovelty", PrototypicalNetworksNovelty(model, use_normcorr=1)],
                             #["PrototypicalNetworks", PrototypicalNetworks(model)], No
                             #["MatchingNetworks", MatchingNetworks(model, feature_dimension=feat_dim)], No - special
                             #["TransductiveFinetuning", TransductiveFinetuning(model)],  No - l2
                             #["SimpleShot", SimpleShot(model)], No - too simple
                             ["Finetune", Finetune(model)], 
                             # #["FEAT", FEAT(model)], 
                             ["BD-CSPN", BDCSPN(model)], 
                             #["LaplacianShot", LaplacianShot(model)], No - special
                             # #["PT-MAP", PTMAP(model)], No
                             ["TIM", TIM(model)]
                            ]
    
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
            print("miniImageNet Val dataset")

    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
    )
    
    #test(model, test_set, test_sampler, few_shot_classifier, n_workers)
    if args.learning:     
        resFileName = args.dataset + '_' + args.model + '_' + str(n_way) + 'way_' + str(n_shot) +"shot_novelty_learn.txt"
    else:
        resFileName = args.dataset + '_' + args.model + '_' + str(n_way) + 'way_' + str(n_shot) +"shot_novelty_test.txt"
        
    novelty_th = getLearnedThreshold(args.weights, args.model, args.shot)

    resFile = open(resDir+subDir+resFileName, "w")
    line = "FewShotClassifier,Accuracy,Threshold\n"
    resFile.write(line)   
    for few_shot in few_shot_classifiers:
        print(few_shot[0])
        print("Use softmax", few_shot[1].use_softmax)
        few_shot_classifier = few_shot[1].to(DEVICE)
        accuracy, threshold = test_or_learn(test_set, test_sampler, few_shot_classifier, 
                                            novelty_th, args.novelty, args.learning, 
                                            n_workers, DEVICE)
        line = few_shot[0] + ',' + str(accuracy) + ',' + str(threshold) + '\n'
        resFile.write(line)    
    resFile.close()
    print("Result saved to", resFileName)
    

