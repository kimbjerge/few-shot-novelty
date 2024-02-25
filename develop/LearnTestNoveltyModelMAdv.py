# -*- coding: utf-8 -*-
"""
Modified on Sun Feb 25 11:34:31 2024

@author: Kim Bjerge
"""

import os
import random
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

from PrototypicalNetworksNovelty import PrototypicalNetworksNovelty
from utilsNovelty import evaluate_M_novel, Metrics

from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks, RelationNetworks, MatchingNetworks, TransductiveFinetuning
from easyfsl.methods import SimpleShot, Finetune, FEAT, BDCSPN, LaplacianShot, PTMAP, TIM
from easyfsl.samplers import TaskSampler

from FewShotModelData import EmbeddingsModel, FewShotDataset

from torchvision.models import resnet50 #, ResNet50_Weights
from torchvision.models import resnet34 #, ResNet34_Weights
from torchvision.models import resnet18 #, ResNet18_Weights


def load_model(modelName, num_classes, argsModel, argsWeights):
    
    if argsModel == 'resnet50':
        #print('resnet50')
        #ResNetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        ResNetModel = resnet50(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        feat_dim = 2048
    if argsModel == 'resnet34':
        #print('resnet34')
        ResNetModel = resnet34(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        feat_dim = 512
    if argsModel == 'resnet18':
        #print('resnet18')
        #ResNetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        ResNetModel = resnet18(pretrained=True) # 80.86, 25.6M
        model = EmbeddingsModel(ResNetModel, num_classes, use_fc=False)
        feat_dim = 512
    if argsModel == 'resnet12':
        #print('resnet12')
        model = resnet12(use_fc=False, num_classes=num_classes) #.to(DEVICE)
        feat_dim = 64
    
    if argsWeights == 'ImageNet':
        print('Using pretrained weights with ImageNet dataset')
    else:
        print('Using saved model weights', modelName)
        modelSaved = torch.load(modelName, map_location=torch.device(DEVICE))
        model.load_state_dict(modelSaved.state_dict())

    model.eval()
    model = model.to(DEVICE)
    
    return model, feat_dim


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
            print("miniImageNet Test dataset")
    
    return test_set


def test_or_learn(test_set, test_sampler, few_shot_classifier, 
                  novelty_th, use_novelty, n_way, n_novel, learn_th, n_workers, metric, DEVICE):

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
       
    accuracy, learned_th, avg, std, avg_o, std_o = evaluate_M_novel(few_shot_classifier, 
                                                            test_loader, 
                                                            novelty_th, 
                                                            device=DEVICE,
                                                            tqdm_prefix="Test",
                                                            plt_hist=True,
                                                            use_novelty=use_novelty,
                                                            n_way = n_way,
                                                            metric=metric,
                                                            learn_th=learn_th,
                                                            n_novel=n_novel
                                                            )
    
    print(f"Average accuracy : {(100 * accuracy):.2f} %")
    return accuracy, learned_th, avg, std, avg_o, std_o


#%% MAIN
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    # Arguments to be changed 
    parser.add_argument('--modelDir', default='models') #Directory that contains Ominiglot models
    parser.add_argument('--way', default=5, type=int) # K-way clases used in support set 
    parser.add_argument('--query', default=10, type=int) # Use 10 for Omniglot and 6 for euMoths (limit of 11 species per class)
    parser.add_argument('--shot', default=5, type=int)  # Number of shot used during learning must be 5
    parser.add_argument('--novel', default=1, type=int) # Number of novel classes used during learning
    parser.add_argument('--cosine', default='True', type=bool) # use Euclidian distance when no parameter ''
    parser.add_argument('--device', default='cpu') #cpu or cuda:0-3
    parser.add_argument('--alpha', default=1.0, type=float) # Used to select model _episodic_<alpha*10>

    # Theses arguments must not be changed and will be updated based on the model name
    parser.add_argument('--model', default='') #resnet12 (Omniglot), resnet18, resnet34, resnet50, Must be empty
    parser.add_argument('--weights', default='') #ImageNet, mini_imagenet, euMoths, CUB, Omniglot, Must be empty
    parser.add_argument('--dataset', default='') #miniImagenet, euMoths, CUB, Omniglot, Must be empty
        
    # Theses arguments must not be changed and will be updated during learning and testing
    parser.add_argument('--novelty', default='', type=bool) #default false when no parameter - automatic False when learning True
    parser.add_argument('--learning', default='', type=bool) #default false when no parameter - learn threshold for novelty detection
    args = parser.parse_args()
 
    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    n_workers = 16
    
    resDir = "./result/"
    dataDirMiniImageNet = "./data/mini_imagenet"
    dataDirEuMoths = "./data/euMoths"
    dataDirCUB = "./data/CUB"
    dataDirOmniglot = "./data/Omniglot"
    subDir = "test/"

    if os.path.exists(resDir+subDir) == False:
        os.mkdir(resDir+subDir)
        print("Create result directory", resDir+subDir)

    #%% Create model and prepare for training
    DEVICE = args.device
    
    similarityName = ""
    if args.cosine:
        similarity_param = 1 # Use cosine similarity
    else:
        similarity_param = 3 # Use euclidian distance
        similarityName = "Euclidean"

    alphaStr = 'episodic_' + str(int(args.alpha*10)) + '_'
    print(alphaStr)
    for modelName in os.listdir(args.modelDir):
        if ('.pth' in modelName) and (alphaStr in modelName):
            modelNameSplit = modelName.split('_')
            #if args.model == '':
            args.model = modelNameSplit[0].lower()
                
            if modelNameSplit[2] == 'imagenet':
                nameData = "miniImagenet"
                nameWeights = "mini_imagenet"
                alpha_idx = 4
            else:
                nameData = modelNameSplit[1]
                nameWeights = nameData
                alpha_idx = 3
                
            #if args.weights == '':
            args.weights = nameWeights
            #if args.dataset == '':
            args.dataset = nameData

            args.alpha = int(modelNameSplit[alpha_idx])/10
            
            print(args)
        
            if args.model == 'resnet12':
                image_size = 28 # Omniglot dataset
            else:
                image_size = 224 # ResNet euMoths
                          
            num_classes = 100  
            if args.weights == 'CUB':
                num_classes = 140  
            if args.weights == 'Omniglot':
                num_classes = 3856  
            if args.weights == 'mini_imagenet':
                num_classes = 60
            
            model, feat_dim = load_model(args.modelDir + '/' +modelName, num_classes, args.model, args.weights)

            n_query = args.query

            #%% Learning
            args.learning = True
            args.novelty = False
            n_shot = args.shot
            n_way = args.way
            n_novel = args.novel
            n_test_tasks = 500 # test tasks - suport + query sets
               
            test_set = load_test_dataset(args.dataset, args.learning)
            
            test_sampler = TaskSampler(
                test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
            )
                                  
            resFileName = args.model + '_' + args.dataset + "_M_novelty_learn.txt"
            line = "ModelDir,ModelName,FewShotClassifier,Way,Shot,Query,Novel,Accuracy,BayesThreshold,Average,Std,AverageOutlier,StdOutlier,MeanBetween\n"
            if os.path.exists(resDir+subDir+resFileName):
                resFile = open(resDir+subDir+resFileName, "a")
            else:
                resFile = open(resDir+subDir+resFileName, "w")
                print(line)
                resFile.write(line)
                resFile.flush()                 
            
            novelty_th = 0.8  # Not used during learning  
            
            few_shot_classifier = PrototypicalNetworksNovelty(model, use_normcorr=similarity_param).to(DEVICE) # Euclidian
            accuracy, threshold, avg, std, avg_o, std_o  = test_or_learn(test_set, test_sampler, few_shot_classifier, 
                                                                         novelty_th, args.novelty, n_way, n_novel, args.learning, 
                                                                         n_workers, None, DEVICE)
            
            line = args.modelDir + ',' + modelName + ',' + "Prototypical" + similarityName + ',' + str(n_way) + ','  
            line += str(n_shot) + ','  + str(n_query) + ',' + str(n_novel) + ',' + str(accuracy) + ',' 
            line += str(threshold) + ',' + str(avg) + ',' + str(std) + ',' + str(avg_o) + ',' + str(std_o) + ',' + str(abs(avg-avg_o)) + '\n'
            print(line)
            resFile.write(line)        
            resFile.close()
            print("Result saved to", resFileName)
                

            #%% Testing
            args.learning = False
            n_test_tasks = 500 # test tasks - suport + query sets

            resFileName =  args.model + '_' +  args.dataset + "_M_novelty_test.txt"
            line = "ModelDir,Model,FewShotClassifier,Novelty,Way,Shot,Query,Novel,Accuracy,Precision,Recall,F1,TP,FP,FN,Method,Threshold,Alpha,ModelName\n"
            if os.path.exists(resDir+subDir+resFileName):
                resFile = open(resDir+subDir+resFileName, "a")
            else:
                resFile = open(resDir+subDir+resFileName, "w")
                print(line)
                resFile.write(line)
                resFile.flush()                 

            novelty_th = threshold # Use learned threshold

            few_shot_classifiers =  [ 
                                     ["Prototypical", PrototypicalNetworksNovelty(model, use_normcorr=similarity_param)], 
                                     #["PrototypicalNetworks", PrototypicalNetworks(model)], #OK - Euclidian only
                                     #["RelationNetworks", RelationNetworks(model, feature_dimension=3)], No
                                     #["MatchingNetworks", MatchingNetworks(model, feature_dimension=feat_dim)], No - special
                                     #["TransductiveFinetuning", TransductiveFinetuning(model)],  No - l2
                                     #["SimpleShot", SimpleShot(model)], No - too simple
                                     #["FEAT", FEAT(model)], - error few-shot and novelty
                                     #["LaplacianShot", LaplacianShot(model)], No - special
                                     #["PT-MAP", PTMAP(model)], No
                                     # ["Finetune", Finetune(model)], #OK
                                     # ["BD-CSPN", BDCSPN(model)], #OK
                                     # ["TIM", TIM(model)] #OK
                                    ]
                
            test_set = load_test_dataset(args.dataset, args.learning)
            
            for n_shot in [5, 1]: # Test with 5 and 1 shot               
                for use_novelty in [True, False]: # Test with and without novelty
                    #use_novelty = True
                    print("Testing with n-shot", n_shot, "novelty", use_novelty)
                    args.novelty = use_novelty
                    if args.novelty:
                        n_novel = args.novel 
                    else:
                        n_novel = 0
                
                    test_sampler = TaskSampler(
                        test_set, n_way=(n_way+n_novel), n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
                    )
                                      
                    for few_shot in few_shot_classifiers:
                        print(few_shot[0])
                        #print("Use softmax", few_shot[1].use_softmax)
                        few_shot_classifier = few_shot[1].to(DEVICE)
                        metric = Metrics()
                        accuracy, threshold, avg, std, avg_o, std_o = test_or_learn(test_set, test_sampler, few_shot_classifier, 
                                                                                    novelty_th, args.novelty, n_way, n_novel, args.learning, 
                                                                                    n_workers, metric, DEVICE)
                        
                        line = args.modelDir + ',' + args.model + ',' + few_shot[0]  + similarityName + ',' + str(args.novelty) + ',' 
                        line += str(n_way) + ','  + str(n_shot) + ','  + str(n_query) + ','  + str(n_novel) + ',' 
                        line += str(accuracy) + ',' + str(metric.precision())  + ',' + str(metric.recall()) + ',' + str(metric.f1score()) + ','
                        line += str(metric.TP()) + ',' + str(metric.FP()) + ',' + str(metric.FN()) + ','
                        line += 'Bayes' + ',' + str(threshold) + ',' + str(args.alpha) + ',' + args.modelDir + '/' + modelName +  '\n'
                        print(line)
                        resFile.write(line)    
                        resFile.flush()
                    
            resFile.close()
            print("Result saved to", resFileName)