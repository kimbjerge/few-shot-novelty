"""
Run this script to prepare the euMoths dataset.

This script uses the 200 classes of 11 images each. The exact images used are 
downloaded from the link (https://inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/)

"""
import numpy as np
import os
import json

DATA_PATH = "../data"


# Find class identities
classes = []
for species in os.listdir(DATA_PATH + '/mini_imagenet/images/'):
    classes.append(species)

classes = list(set(classes))
print(classes)

# Train/val/test split
np.random.seed(0)
np.random.shuffle(classes)
train_classes, val_classes, test_classes = classes[:60], classes[60:80], classes[80:] # Split 60 - 20 - 20

train_roots = []
val_roots = []
test_roots = []
# Move images to correct location
for species in os.listdir(DATA_PATH + '/mini_imagenet/images/'):
    class_name = species
    if class_name in train_classes:
        train_roots.append('./data/mini_imagenet/images/'+class_name)
    if class_name in val_classes:
        val_roots.append('./data/mini_imagenet/images/'+class_name)
    if class_name in test_classes:
        test_roots.append('./data/mini_imagenet/images/'+class_name)
                
train_json = {}
train_json['class_names'] = train_classes
train_json['class_roots'] = train_roots
val_json = {}
val_json['class_names'] = val_classes
val_json['class_roots'] = val_roots
test_json = {}
test_json['class_names'] = test_classes
test_json['class_roots'] = test_roots

with open("./data/mini_imagenet/train.json", "w") as outfile:
    json.dump(train_json, outfile)
with open("./data/mini_imagenet/val.json", "w") as outfile:
    json.dump(val_json, outfile)
with open("./data/mini_imagenet/test.json", "w") as outfile:
    json.dump(test_json, outfile)
    
