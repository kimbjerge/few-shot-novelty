# -*- coding: utf-8 -*-
"""
Modified on Sun Feb 25 11:34:31 2024

@author: Kim Bjerge
"""

import random
import argparse
import numpy as np
import torch
from statistics import mean
from torch import nn
from tqdm import tqdm
from datetime import datetime

from PrototypicalNetworksNovelty import PrototypicalNetworksNovelty

from easyfsl.modules import resnet12
from easyfsl.methods import FewShotClassifier
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate

from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet50 #, ResNet50_Weights
from torchvision.models import resnet34 #, ResNet34_Weights
from torchvision.models import resnet18 #, ResNet18_Weights


from FewShotModelData import EmbeddingsModel, FewShotDataset


#%% Episodic training      
def train_episodic_epoch(lossFunction, 
                         model: FewShotClassifier, 
                         data_loader: DataLoader, 
                         optimizer: Optimizer,
                         slossFunc,
                         alpha,
                         cosine):
    all_loss = []
    all_closs = []
    all_sloss = []
    all_scatter_between = []
    model.train()
    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:

            optimizer.zero_grad()
            model.process_support_set(
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )

            classification_scores = model(query_images.to(DEVICE))
            
            closs = lossFunction(classification_scores, query_labels.to(DEVICE))

            if slossFunc == "Multi" or slossFunc == "MultiAlt":
                ScatterBetween, ScatterWithin, sloss = model.multivariantScatterLoss()
                if slossFunc == "MultiAlt":
                    sloss = 100000/ScatterBetween   
            else:
                correct_episodes = classification_scores[torch.max(classification_scores, 1)[1] == query_labels.to(DEVICE)]
                correct_scores = correct_episodes.max(1)[0]
                correct_pred_idx = correct_episodes.max(1)[1]            
                
                #Select scores part of correct predicitons that don't belong to the query label
                num_rows = correct_episodes.shape[0]
                num_cols = correct_episodes.shape[1]
                wrong_scores = torch.empty(num_rows*(num_cols-1)).to(DEVICE)
                idx = 0
                for i in range(num_rows):
                    for j in range(num_cols):
                        if j != correct_pred_idx[i]:
                            wrong_scores[idx]=correct_episodes[i][j]
                            idx += 1
                
                ScatterWithin = 1 # Mean only
                if slossFunc == "Std": # Mean and standard deviation
                    ScatterWithin = correct_scores.std() + wrong_scores.std()
                
                ScatterBetween = abs(correct_scores.mean() - wrong_scores.mean())
                sloss = ScatterWithin/ScatterBetween # Minimize scatter within related to scatter between      
 
            if torch.isnan(sloss): # Handling division with zero
                print("sloss nan")
                loss = closs 
            else:
                loss = alpha*sloss + (1-alpha)*closs
                
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())
            all_closs.append(closs.item())
            all_sloss.append(sloss.item())
            all_scatter_between.append(ScatterBetween.item())

            tqdm_train.set_postfix( loss="{:.4f}".format(mean(all_loss)), 
                                    closs="{:.4f}".format(mean(all_closs)), 
                                    sloss="{:.4f}".format(mean(all_sloss)) )

    return mean(all_loss), mean(all_closs), mean(all_sloss), mean(all_scatter_between)


def CosineEmbeddingLoss(scores, labels, margin=0.6):
# https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss    
  
    # 1 - score if y = 1
    # max(0, score) if y = -1
    
    #print(scores)
    #print(labels)
    predicted_idx = scores.max(1)[1]
    loss_sum = 0
    for idx in range(len(labels)):
        score = scores[idx][labels[idx]] 
        if labels[idx] == predicted_idx[idx]:
            loss_sum += 1 - score # y = 1, correct predicted
        else:
            loss_sum += max([0, score - margin]) # y = -1, wrongly predicted
        
    return loss_sum


def episodicTrain(modelName, train_loader, val_loader, few_shot_classifier, 
                  m1=500, m2=1000, n_epochs=1500, alpha=0.1, slossFunc="Mean", 
                  cosine=False, learnRate=0.1, pretrained=False):
    
    if cosine:
        entropyLossFunction = CosineEmbeddingLoss
        #entropyLossFunction = nn.CrossEntropyLoss()
        print("CosineEmbeddingLoss, margin = 0.6")
    else:
        entropyLossFunction = nn.CrossEntropyLoss()
        print("CrossEntropyLoss")
    
    #scheduler_milestones = [10, 30]
    if n_epochs < 1000:
        scheduler_milestones = [60, 120] # From scratch with 200 epochs
    else:
        scheduler_milestones = [m1, m2] # From scratch with 1500 epochs
        
    scheduler_gamma = 0.1
    learning_rate = learnRate # 1e-2
    
    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    #tb_logs_dir = Path("./logs")   
    #tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
    log_dir = '-' + modelName.split('/')[2].replace(".pth", "")
    tb_writer = SummaryWriter(comment=log_dir)

    # Train model
    best_state = few_shot_classifier.state_dict()
    best_loss = 1000.0
    best_validation_accuracy = 0.0
    best_scatter_between = 0.0
    best_epoch = 0
    for epoch in range(n_epochs):
        if epoch < 0:
            alphaUsed = 0.0 # Prioritize cross entropy loss in start
        else:
            alphaUsed = alpha
        print(f"Epoch {epoch} Alpha {alphaUsed}")
        average_loss, average_closs, average_sloss, average_scatter_between = train_episodic_epoch(entropyLossFunction, 
                                                                                                   few_shot_classifier, train_loader, 
                                                                                                   train_optimizer, slossFunc, 
                                                                                                   alphaUsed, cosine)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )
        
        if pretrained:    
            if best_loss > average_loss: 
                best_epoch = epoch+1
                best_loss = average_loss
                best_validation_accuracy = validation_accuracy
                best_scatter_between = average_scatter_between
                best_state = few_shot_classifier.state_dict()
                torch.save(few_shot_classifier.backbone, modelName)
                print(f"Lowest loss model saved with accuracy {(best_validation_accuracy):.4f} and loss {(best_loss):.4f}", modelName)
        else:
            if validation_accuracy > best_validation_accuracy:
                best_epoch = epoch+1
                best_loss = average_loss
                best_validation_accuracy = validation_accuracy
                best_scatter_between = average_scatter_between
                best_state = few_shot_classifier.state_dict()
                torch.save(few_shot_classifier.backbone, modelName)
                print(f"Best model saved with accuracy {(best_validation_accuracy):.4f} and loss {(best_loss):.4f}", modelName)
                
        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Train/closs", average_closs, epoch)
        tb_writer.add_scalar("Train/sloss", average_sloss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
    
        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    print(f"Best validation accuracy {(best_validation_accuracy):.4f} with loss {(best_loss):.4f} after epochs", best_epoch)

    return best_state, few_shot_classifier, best_epoch, best_validation_accuracy, best_scatter_between, best_loss


#%% Few shot testing of model        
def test(model, test_loader, few_shot_classifier, n_workers, DEVICE):
    
    model.eval()

    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE, tqdm_prefix="Test")
    return accuracy

#%% Saving result to file  
def saveArgs(modelName, args, best_epoch, valAccuracy, testAccuracy, scatterBetween, bestLoss):
    
    with open(modelName.replace('.pth', '.txt'), 'w') as f:
        line = "model,dataset,mode,cosine,epochs,m1,m2,slossFunc,alpha,pretrained,learnRate,device,trainTasks,"
        line += "valTasks,way,query,shot,bestEpoch,valAccuracy,testAccuracy,meanBetween,trainLoss,modelName\n"
        f.write(line)
        line = args.model + ','
        line += args.dataset + ','
        line += args.mode + ',' 
        line += str(args.cosine) + ',' 
        line += str(args.epochs) + ',' 
        line += str(args.m1) + ','
        line += str(args.m2)  + ','
        line += args.slossFunc + ',' 
        line += str(args.alpha) + ','
        line += str(args.pretrained) + ','
        line += str(args.learnRate) + ','
        line += args.device + ','
        line += str(args.tasks) + ',' 
        line += str(args.valTasks) + ','
        line += str(args.way) + ','
        line += str(args.query) + ','
        line += str(args.shot) + ','
        line += str(best_epoch) + ','
        line += str(valAccuracy) + ','
        line += str(testAccuracy) + ','
        line += str(scatterBetween) + ','
        line += str(bestLoss) + ','
        line += modelName + '\n'
        print(line)
        f.write(line)        

#%% MAIN
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet12') #resnet12, resnet18, resnet34, resnet50
    parser.add_argument('--dataset', default='Omniglot') #euMoths, CUB, Omniglot (resnet12), mini_imagenet
    parser.add_argument('--mode', default='episodic') # episodic only (classic removed in this version)
    parser.add_argument('--cosine', default='', type=bool) # default use Euclidian distance when no parameter ''
    parser.add_argument('--epochs', default=2, type=int) # epochs
    parser.add_argument('--m1', default=120, type=int) # learning rate scheduler for milstone 1 (epochs)
    parser.add_argument('--m2', default=190, type=int) # learning rate scheduler for rate milstone 2 (epochs)
    parser.add_argument('--slossFunc', default='Std') # scatter loss function with univariant using standard deviation (Std) or only mean (Mean), multivariate (Multi)
    parser.add_argument('--alpha', default=1.0, type=float) # alpha parameter for sloss function (0-1)
    parser.add_argument('--pretrained', default='', type=bool) # default pretrained weigts is false
    parser.add_argument('--device', default='cpu') # training on cpu or cuda:0-3
    parser.add_argument('--tasks', default='200', type=int) # training tasks per epoch
    parser.add_argument('--valTasks', default='100', type=int) # tasks used for validation
    parser.add_argument('--way', default='5', type=int) # K-ways for episodic training and few-shot validation
    parser.add_argument('--query', default='10', type=int) # Q-query for episodic training and few-shot validation
    parser.add_argument('--shot', default='5', type=int) # N-shot for episodic training always 5-shot
    parser.add_argument('--learnRate', default='0.05', type=float) # learn rate for episodic and classic training
    args = parser.parse_args()
 
    dataDir = './data/' + args.dataset
    image_size = 224 # ResNet euMoths and CUB
    n_epochs = args.epochs # ImageNet pretrained weights - finetuning
    
    if  args.model == 'resnet12':
        
        if args.model == 'CUB':
            image_size = 84 # CUB dataset and resnet12
        
        if args.dataset == 'Omniglot':
            image_size = 28 # Omniglot dataset

    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    n_workers = 12
 
    n_way = args.way # 5 or 20 paper did
    n_shot = args.shot # For episodic training use 5 shot
    n_query = args.query
    n_tasks_per_epoch = args.tasks
    n_validation_tasks = args.valTasks
    n_test_tasks = 200 # For final test, result saved to result CSV file
   
    #%% Creating dataloaders for training and validation
    # Training dataset
    train_set = FewShotDataset(split="train",  image_size=image_size, root=dataDir, training=True)    
        
    train_sampler = TaskSampler(
        train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )    
    n_use_fc = False
   
    # Validation dataset
    val_set = FewShotDataset(split="val",  image_size=image_size, root=dataDir, training=False)
    val_sampler = TaskSampler(
        val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )
    
    #%% Create model and prepare for training
    DEVICE = args.device
    
    num_classes = len(set(train_set.get_labels()))
    print("Training classes", num_classes)
    print("Validation classes", len(set(val_set.get_labels())))
    
    now = datetime.now()
    dateTime = now.strftime("%m%d_%H%M%S")
    if args.model == 'resnet50':
        print('resnet50')
        NetModel = resnet50(pretrained=args.pretrained).to(DEVICE)    
        modelName = "./models/Resnet50_" + args.dataset + '_' + args.mode + '_' + str(int(args.alpha*10)) + '_' + dateTime + "_AdvLoss.pth"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc)
        
    if args.model == 'resnet34':
        print('resnet34')
        NetModel = resnet34(pretrained=args.pretrained).to(DEVICE)
        modelName = "./models/Resnet34_" + args.dataset + '_' + args.mode + '_' + str(int(args.alpha*10))  + '_' + dateTime + "_AdvLoss.pth"   
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc)
        
    if args.model == 'resnet18':
        print('resnet18')
        NetModel = resnet18(pretrained=args.pretrained).to(DEVICE) 
        modelName = "./models/Resnet18_" + args.dataset + '_' + args.mode  + '_' + str(int(args.alpha*10)) + '_' + dateTime + "_AdvLoss.pth"
        model = EmbeddingsModel(NetModel, num_classes, use_softmax=False, use_fc=n_use_fc)
        
    if args.model == 'resnet12':
        print('resnet12')
        modelName = "./models/Resnet12_" + args.dataset + '_' + args.mode + '_' + str(int(args.alpha*10)) + '_' + dateTime +"_AdvLoss.pth"  
        # This model is not retrained, but trained from scratch
        model = resnet12(use_fc=n_use_fc, num_classes=num_classes).to(DEVICE)
        
    model = model.to(DEVICE)
    print("Saving model as", modelName)
    saveArgs(modelName, args, 0, 0, 0, 0, 0)

    if args.cosine:
        few_shot_classifier = PrototypicalNetworksNovelty(model).to(DEVICE)
        print("Use prototypical network with cosine distance to train and validate")
    else:
        #few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)
        few_shot_classifier = PrototypicalNetworksNovelty(model, use_normcorr=3).to(DEVICE)
        print("Use prototypical network with euclidian distance to train and validate")
    
    best_scatter_between = 0
    best_loss = 0

    #%% Episodic training of model using train_loader and val_loader
    print("Episodic training epochs", n_epochs)
    best_state, model, best_epoch, best_accuracy, best_scatter_between, best_loss = episodicTrain(modelName, 
                                                                                                  train_loader, val_loader, 
                                                                                                  few_shot_classifier, 
                                                                                                  m1=args.m1, m2=args.m2, 
                                                                                                  n_epochs=n_epochs, alpha=args.alpha, 
                                                                                                  slossFunc=args.slossFunc,
                                                                                                  cosine=args.cosine,
                                                                                                  learnRate=args.learnRate,
                                                                                                  pretrained=args.pretrained)
    few_shot_classifier.load_state_dict(best_state)

    #%% Final testing the best model on test dataset
    test_set = FewShotDataset(split="test", image_size=image_size, root=dataDir, training=False)
    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    accuracy = test(model, test_loader, few_shot_classifier, n_workers, DEVICE)
    
    saveArgs(modelName, args, best_epoch, best_accuracy, accuracy, best_scatter_between, best_loss)

    textLine = f"Accuracy val/test : {(100 * best_accuracy):.2f}%/{(100 * accuracy):.2f}%," + args.model + "," + args.dataset 
    textLine += "," + args.slossFunc + ',' + str(args.alpha) + "," + str(best_epoch) + "," + f"{(best_loss):.4f}," +  modelName + '\n'
    print(textLine)
    with open('ResultTrainAdvLoss.txt', 'a') as f:
        f.write(textLine)
