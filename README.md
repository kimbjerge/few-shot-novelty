# few-shot-novelty
This project contains work in progrees with Python code for few-shot-learning with novelty detection. 

The problem addressed with this research is to solve the extended few-shot learning problem with queries of images of both known and unknown images. 
A support set contains N classes with each K shots of images and a query set that contains Q images to be classified as classes inside or outside of the support set.
An image in the query set that does not belong to any of the N classes in the support should be detected as a novelty. 

 
## Python environment installations 

Install the Python libraries as specified in dev_requirements.txt.

The easy-few-shot-learning (easyfsl) framework has been used to boost our experiments with few-shot image classification.
The framework contains libraries for 11 few-shot learning methods, handling of support and query data and modules for resnet12 backend with episodic training. 

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



