# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:58:35 2023

@author: Kim Bjerge
"""

from pathlib import Path
import random
import argparse
import numpy as np
import torch
from statistics import mean
from torch import nn
from tqdm import tqdm

#from easyfsl.datasets import CUB
#from easyfsl.datasets import EasySet

from PrototypicalNetworksNovelty import PrototypicalNetworksNovelty

from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks, FewShotClassifier, Finetune
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate

from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet50 #, ResNet50_Weights
from torchvision.models import resnet34 #, ResNet34_Weights
from torchvision.models import resnet18 #, ResNet18_Weights

#from torchvision.models.efficientnet import efficientnet_b3 #, EfficientNet_B3_Weights
#from torchvision.models.efficientnet import efficientnet_b4 #, EfficientNet_B4_Weights
#from torchvision.models.efficientnet import efficientnet_b7 #, EfficientNet_B7_Weights

from FewShotModelData import EmbeddingsModel, FewShotDataset


#%% Classical training      
def train_epoch(entropyLossFunction: nn.CrossEntropyLoss, 
                   model_: nn.Module, 
                   data_loader: DataLoader, 
                   optimizer: Optimizer):
    
    all_loss = []
    model_.train()
    with tqdm(data_loader, total=len(data_loader), desc="Training") as tqdm_train:
        for images, labels in tqdm_train:
            optimizer.zero_grad()

            predictions = model_(images.to(DEVICE))
            loss = entropyLossFunction(predictions, labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss="{}".format(mean(all_loss)))

    return mean(all_loss)


def classicTrain(model, modelName, train_loader, val_loader, few_shot_classifier,  n_epochs=200):

    #scheduler_milestones = [3, 6]
    #scheduler_milestones = [500, 1000] # From scratch with 500 epochs
    scheduler_milestones = [70, 140] # From scratch with 200 epochs
    scheduler_gamma = 0.1
    #learning_rate = 5e-4 # 1e-1 - without pretrained weights 5e-4 - with pretrained weights
    learning_rate = 0.1 # 1e-1 - without pretrained weights 5e-4 - with pretrained weights
    tb_logs_dir = Path("./logs")
   
    entropyLossFunction = nn.CrossEntropyLoss()
   
    train_optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #train_optimizer = Adam(model.parameters(), lr=learning_rate) # Not working
    
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )
    
    tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
    
    best_state = model.state_dict()
    best_validation_accuracy = 0.0
    validation_frequency = 5
    best_epoch = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = train_epoch(entropyLossFunction, model, train_loader, train_optimizer)
    
        if epoch % validation_frequency == validation_frequency - 1:
    
            # We use this very convenient method from EasyFSL's ResNet to specify
            # that the model shouldn't use its last fully connected layer during validation.
            model.set_use_fc(False)
            model.eval()
            validation_accuracy = evaluate(few_shot_classifier, val_loader, 
                                           device=DEVICE, tqdm_prefix="Validation")
            model.set_use_fc(True)
    
            if validation_accuracy > best_validation_accuracy:
                best_epoch = epoch
                best_validation_accuracy = validation_accuracy
                best_state = model.state_dict()
                print("Ding ding ding! We found a new best model!")
                torch.save(model, modelName)
                print("Best model saved", modelName)
    
            tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
    
        tb_writer.add_scalar("Train/loss", average_loss, epoch)
    
        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()
    
    print("Best validation accuracy after epoch", best_validation_accuracy, best_epoch)
    
    return best_state, model
    
#%% Episodic training      
def train_episodic_epoch(entropyLossFunction: nn.CrossEntropyLoss, 
                         model: FewShotClassifier, 
                         data_loader: DataLoader, 
                         optimizer: Optimizer):
    
    all_loss = []
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

            loss = entropyLossFunction(classification_scores, query_labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)

def episodicTrain(modelName, train_loader, val_loader, few_shot_classifier,  n_epochs=200):
    
    entropyLossFunction = nn.CrossEntropyLoss()
    
    #scheduler_milestones = [10, 30]
    scheduler_milestones = [500, 1000] # From scratch with 1500 epochs
    scheduler_gamma = 0.1
    learning_rate = 1e-1 # 1e-2
    tb_logs_dir = Path("./logs")
    
    
    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

    # Train model
    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0
    best_epoch = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = train_episodic_epoch(entropyLossFunction, few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )
    
        if validation_accuracy > best_validation_accuracy:
            best_epoch = epoch
            best_validation_accuracy = validation_accuracy
            best_state = few_shot_classifier.state_dict()
            print("Ding ding ding! We found a new best model!")
            torch.save(few_shot_classifier, modelName)
            print("Best model saved", modelName)
    
        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
    
        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    print("Best validation accuracy after epoch", best_validation_accuracy, best_epoch)

    return best_state, few_shot_classifier


#%% Few shot testing of model        
def test(model, test_loader, few_shot_classifier, n_workers, DEVICE):
    
    model.eval()

    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE, tqdm_prefix="Test")
    print(f"Average accuracy : {(100 * accuracy):.2f} %")


#%% MAIN
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18') #resnet12, resnet18, resnet34, resnet50
    parser.add_argument('--dataset', default='CUB') #euMoths, CUB, Omniglot
    parser.add_argument('--mode', default='classic') #classic, episodic
    parser.add_argument('--cosine', default='', type=bool) # Default use Euclidian distance when no parameter ''
    parser.add_argument('--epochs', default=1, type=int) #epochs
    args = parser.parse_args()
       
    dataDir = './data/' + args.dataset
    image_size = 224 # ResNet euMoths and CUB
    n_epochs = args.epochs # ImageNet pretrained weights - finetuning
    
    if  args.model == 'resnet12':
        
        if args.model == 'CUB':
            n_epochs = 200 # Trained from scratch
            image_size = 84 # CUB dataset
        
        if args.dataset == 'Omniglot':
            #n_epochs = 15 # Trained from scratch - episodic use 50
            image_size = 28 # Omniglot dataset
            
    #image_size = 300 # EfficientNet B3
    #image_size = 380 # EfficientNet B4
    #image_size = 600 # EfficientNet B7

    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #batch_size = 256
    batch_size = 128
    n_workers = 6
 
    n_way = 5
    n_shot = 5 # Use 3 shot for validation
    n_query = 6
    n_tasks_per_epoch = 50
    n_validation_tasks = 30
    n_test_tasks = 200
   
    # Training dataset
    train_set = FewShotDataset(split="train",  image_size=image_size, root=dataDir, training=True)    
    if args.mode == 'classic':
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            shuffle=True,
        )
        n_use_fc = True
        
    if args.mode == 'episodic': # Use task sample for episodic training
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
    DEVICE = "cuda:1"
    #DEVICE = "cpu"
    
    num_classes = len(set(train_set.get_labels()))
    print("Training classes", num_classes)
    print("Validation classes", len(set(val_set.get_labels())))
    
    # Stable models https://pytorch.org/vision/stable/models.html   # Top 1, Accuracy
    #NetModel = efficientnet_b7(pretrained=True)                 # 84.122, 66.3M   
    #NetModel = efficientnet_b4(pretrained=True)                 # 83.384, 19.3M   
    #NetModel = efficientnet_b3(pretrained=True)                 # 82.003, 12.3M   
    #modelName = "./models/EffnetB4_euMoths_model.pth"
    
    if args.model == 'resnet50':
        print('resnet50')
        #NetModel = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.858, 25.6M
        NetModel = resnet50(pretrained=True).to(DEVICE)    
        modelName = "./models/Resnet50_" + args.dataset + '_' + args.mode + ".pth"
        model = EmbeddingsModel(NetModel, num_classes, use_fc=n_use_fc)
        
    if args.model == 'resnet34':
        print('resnet34')
        #NetModel = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) # 73.314, 21.8M
        NetModel = resnet34(pretrained=True).to(DEVICE)
        modelName = "./models/Resnet34_" + args.dataset + '_' + args.mode + ".pth"   
        model = EmbeddingsModel(NetModel, num_classes, use_fc=n_use_fc)
        
    if args.model == 'resnet18':
        print('resnet18')
        #NetModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # 69.758, 11.7M
        NetModel = resnet18(pretrained=False).to(DEVICE) 
        modelName = "./models/Resnet18_" + args.dataset + '_' + args.mode + ".pth"
        model = EmbeddingsModel(NetModel, num_classes, use_fc=n_use_fc)
        
    if args.model == 'resnet12':
        print('resnet12')
        modelName = "./models/Resnet12_" + args.dataset + '_' + args.mode + ".pth"  
        # This model is not retrained, but trained from scratch
        model = resnet12(use_fc=n_use_fc, num_classes=num_classes).to(DEVICE)
        
    model = model.to(DEVICE)
    print("Saving model as", modelName)

    if args.cosine:
        few_shot_classifier = PrototypicalNetworksNovelty(model).to(DEVICE)
        print("Use prototypical network with cosine distance to validate")
    else:
        few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)
        print("Use prototypical network with euclidian distance to validate")
    
    if args.mode == 'classic':
        print("Classic training epochs", n_epochs)
        best_state, model = classicTrain(model, modelName, train_loader, val_loader, few_shot_classifier, n_epochs=n_epochs)
        model.set_use_fc(False)       

    if args.mode == 'episodic':
        print("Episodic training epochs", n_epochs)
        best_state, model = episodicTrain(modelName, train_loader, val_loader, few_shot_classifier, n_epochs=n_epochs)
    
    model.load_state_dict(best_state)
    
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
    test(model, test_loader, few_shot_classifier, n_workers, DEVICE)
    
