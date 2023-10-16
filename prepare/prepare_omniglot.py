"""
Run this script to prepare the Omniglot dataset.

This script uses the 200 classes of 11 images each. The exact images used are 
downloaded from the link (https://inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/)

"""
import numpy as np
import os
import json

DATA_PATH = "../data"

# Find class identities
train_classes = []
for alphabet in os.listdir(DATA_PATH + '/Omniglot/images_background/'):
    for character in os.listdir(DATA_PATH + '/Omniglot/images_background/' + alphabet):
        train_classes.append(alphabet+'/'+character)

train_roots = []
# Move images to correct location
for alphabet in os.listdir(DATA_PATH + '/Omniglot/images_background/'):
    for character in os.listdir(DATA_PATH + '/Omniglot/images_background/' + alphabet):
        class_name = alphabet+'/'+character
        if class_name in train_classes:
            train_roots.append('./data/Omniglot/images_background/'+class_name)
        
        
classes_eval = []
for alphabet in os.listdir(DATA_PATH + '/Omniglot/images_evaluation/'):
    for character in os.listdir(DATA_PATH + '/Omniglot/images_evaluation/' + alphabet):
        classes_eval.append(alphabet+'/'+character)
    
classes_eval = list(set(classes_eval))
print(classes_eval)

# Train/val/test split
np.random.seed(0)
np.random.shuffle(classes_eval)
val_classes, test_classes = classes_eval[:40], classes_eval[40:80]

val_roots = []
test_roots = []
# Move images to correct location
for alphabet in os.listdir(DATA_PATH + '/Omniglot/images_evaluation/'):
    for character in os.listdir(DATA_PATH + '/Omniglot/images_evaluation/' + alphabet):
        class_name = alphabet+'/'+character
        if class_name in val_classes:
            val_roots.append('./data/Omniglot/images_evaluation/'+class_name)
        if class_name in test_classes:
            test_roots.append('./data/Omniglot/images_evaluation/'+class_name)
                
train_json = {}
train_json['class_names'] = train_classes
train_json['class_roots'] = train_roots
val_json = {}
val_json['class_names'] = val_classes
val_json['class_roots'] = val_roots
test_json = {}
test_json['class_names'] = test_classes
test_json['class_roots'] = test_roots

with open("./data/Omniglot/train.json", "w") as outfile:
    json.dump(train_json, outfile)
with open("./data/Omniglot/val.json", "w") as outfile:
    json.dump(val_json, outfile)
with open("./data/Omniglot/test.json", "w") as outfile:
    json.dump(test_json, outfile)
    
