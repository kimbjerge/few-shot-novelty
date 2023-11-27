# few-shot-novelty
This project contains work in progrees with code for few-shot-learning with novelty detection. 

The problem addressed with this research is to solve the extended few-shot learning problem with queries of images of both known and unknown images. 
A support set contains N-classes (N-way) with each K-shots of images and a query set that contains Q images to be classified as classes inside or outside of the support set.
An image in the query set that does not belong to any of the N classes in the support should be detected as a novelty. 
 
## Python environment installations 

Install the Python libraries as specified in dev_requirements.txt.

The easy-few-shot-learning (easyfsl) framework has been used to boost our experiments with few-shot image classification.
The framework contains libraries for 11 few-shot learning methods, handling of support and query data and Python code for resnet12 backend with episodic training. 

Install the Python library easyfsl "pip install easyfsl" or use the GitHub:

https://github.com/sicara/easy-few-shot-learning

## Datasets used for training, validation, and testing

A copy of the prepared Omniglot, CU-Birds and EU moths datasets can be downloaded from here:

https://drive.google.com/drive/folders/1xaAJG_-wGpqR0JRUAEjzbcZyS5GxrhNk

The zipped files must be copied and unzipped to the folders: 
- data/Omniglot
- data/CUB
- data/euMoths

### miniImageNet 

This dataset presents a preprocessed version of the miniImageNet benchmark dataset used in few-shot learning.
This version of miniImageNet is not resized to any particular size and is left to be the same size as they are in the ImageNet dataset.

Download and unzip the preprocessed version of the miniImageNet benchmark dataset from:
https://www.kaggle.com/datasets/arjunashok33/miniimagenet

Copy the image files to data/mini_imagenet

The train.json, val.json, and test.json split the dataset into 60 train images, 20 validation images, and 20 test images. 

With prepare/prepare_mini_imagenet.py it is possible to create another split of the miniImageNet dataset.

### Omniglot

The Omniglot data set is designed for developing more human-like learning algorithms. 
It contains 1623 different handwritten characters from 50 different alphabets and is used for few-shot learning research.

Alternatively download and unzip the images_background.zip and images_evaluation.zip from the below GitHub. 
https://github.com/brendenlake/omniglot

Copy the images files to data/Omniglot

The train.json, val.json, and test.json split the dataset into 3856 train images, 40 validation images, and 40 test images.  

With the Python script: prepare/prepare_Omniglot.py it is possible to create a customized split of the Omniglot dataset.

### CU-Birds (CUB)

The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is a widely-used dataset for fine-grained visual categorization task. 
It contains 11,788 images of 200 subcategories belonging to birds.

Alternatively download and extract the dataset from a Github which provides a make download-cub recipe to download and extract the dataset.
See https://github.com/sicara/easy-few-shot-learning

The train.json, val.json, and test.json split the dataset into 140 train images, 30 validation images, and 30 test images.  

### EU moths

This dataset presents a dataset of only 11 samples for each class of 200 classes of moth species.

Alternatively download and unzip the Cropped images of the EU Moths dataset from:
https://inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/

Copy the image files to data/euMoths/images

The train.json, val.json, and test.json split the dataset into 100 train images, 50 validation images, and 50 test images. 

With prepare/prepare_eu_moths.py it is possible to create another split of the EU moths dataset.


## Episodic training

Episodic training for domain generalization is the problem of learning models that generalize to novel testing domains with different statistics and classes than in the set of the known training domains. 
The method learns a model that generalizes well to a novel domain without any knowledge of the novel domain with new classes during episodic model training.
It is also called the meta-learning paradigm, here we have a set of tasks to learn in each epoch. 
Each task also called an episode contains a support set of N-classes (N-way) with a K-shot of images for each class. 
A query set of images is matched with the support set using a few-shot Protypical network that compares the embeddings from the backbone of the convolutional neural network.
The Prototypical network uses the Euclidian distance as a similarity function during training to find the best matching class in the support set. 
Episodic training can be performed with and without pre-trained weights where the backbone is ResNet18, ResNet34, or ResNet50.

- When training without pre-trained weights the model with the best accuracy is selected and stored.
- When training with pre-trained weights the model with the lowest loss is selected and stored.  

The models and results will be stored in the folder modelsAdv and tensorboard log files are stored in the folder runs.

To view the tensorboard log files write: tensorflow --logdir runs/<model name log dir>


### Omniglot training

To train a model on the Omniglot dataset with ResNet12 without pre-trained weights the Linux bash script/trainOmniglotAdv.sh is provided.
The alpha value is used to prioritize between cross entropy and scatter loss during training. (See paper)
The Linux script contains command arguments used to perform episodic training of models with alpha values ranging from 0.0 - 1.0 as in the example shown below.

python FewShotTrainingAdvLoss.py --model resnet12 --dataset Omniglot --epochs 350 --m1 120 --m2 250 --slossFunc Std --learnRate 0.05 --alpha 0.5 --tasks 200 --valTasks 100 --query 10 --device cuda:0

The folder modelsOmniglotAdvStd4 contains the trained models with files that are generated for every model and contains arguments and results for training.


### CU-Birds and EU moths training with transfer learning

To train models on the CUB and EU-Moths dataset with pretrained weights from ImageNet the backbones ResNet18, ResNet34 and ResNet50 is provided.
It is also possible to train miniImageNet with pre-trained weights, however, since miniImageNet is a subset of ImagneNet it would give unrealistic good results for domain adaptation since the same classes are included during pre-training and validation.

The Linux bash script/trainCUBPreAdv.sh contains command arguments to train with the CU-Birds dataset: 

python FewShotTrainingAdvLoss.py --model resnet18 --dataset CUB --mode episodic --slossFunc Std --alpha 0.5 --m1 120 --m2 190 --epochs 250 --learnRate 0.001 --pretrained True --tasks 500 --valTasks 100 --query 10 --device cuda:0

The linux bash script/traineuMothsPreAdv.sh contains command arguments to train with the EU-Moths dataset: 

python FewShotTrainingAdvLoss.py --model resnet18 --dataset euMoths --mode episodic --alpha 0.5 --m1 120 --m2 190 --epochs 250 --learnRate 0.001 --pretrained True --slossFunc Std --tasks 500 --valTasks 100 --query 6 --device cuda:0

The folder modelsFinalPreAdv contains the trained models with files that are generated for every model and contains arguments and results for training.


### miniImageNet training without pre-trained weights

To train models on the miniImageNet dataset without pre-trained weights the backbones ResNet18, ResNet34, and ResNet50 are provided.

The Linux bash script/trainImageNetAdv.sh contains command arguments to train with the miniImageNet dataset: 

python FewShotTrainingAdvLoss.py --model resnet18 --dataset mini_imagenet --mode episodic --slossFunc Std --alpha 0.5 --m1 120 --m2 190 --epochs 250 --learnRate 0.01 --tasks 500 --valTasks 100 --query 10 --device cuda:0

The folder modelsFinalAdv contains the trained models with files that are generated for every model and contains arguments and results for training.


## Learning and testing

To learn the Bayes threshold for detection of the novel class and test the models. Python code is provided that can be configured with arguments.

The LearnTestNoveltyModelAdv.py will take all models (*.pth) in the directory specified by --modelDir and perform the below operations.

1. The Bayes threshold is learned with few-shot learning (N-way and 5-shot) on the validation dataset
2. A few-shot with novelty is performed on the test dataset (N-way, 1-novelty and 5-shot)
3. few-shot without novelty is performed on the test dataset (N-way and 5-shot)
4. A few-shot with novelty is performed on the test dataset (N-way, 1-novelty, and 1-shot)
5. A few-shot without novelty is performed on the test dataset (N-way and 1-shot)

All results are stored in the folder results/test with the filenames *_learn.txt and *_text.txt


### Learning and test results for 5-ways with 10 images in each query for trained models on Omniglot

python LearnTestNoveltyModelAdv.py --modelDir modelsOmniglotAdvStd4 --way 5 --query 10 --device cuda:0

Pre-processed result files are stored in modelsOmniglotAdvStd4/results-5w and modelsOmniglotAdvStd4/results-20w


### Learning and test results for 5-ways with 6 images in each query for trained models on CUB and euMoths

python LearnTestNoveltyModelAdv.py --modelDir modelsFinalPreAdv --way 5 --query 6 --device cuda:0

There are only 11 images for each class in the EU-Moths dataset therefore with 5-shot, the maximum query of images is 6.

Pre-processed result files are stored in modelsOmniglotAdvStd4/results-5w


### Learning and test results for 5-ways with 10 images in each query for trained models on miniImagenet

python LearnTestNoveltyModelAdv.py --modelDir modelsFinalPreAdv --way 5 --query 10 --device cuda:0


## Printing and plotting results

PlotNoveltyResultsAdv.py and PlotFewShotNoveltyResultsAdv.py are used to read and plot the result files *_text.txt.
