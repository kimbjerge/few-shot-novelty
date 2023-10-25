# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:21:11 2023

@author: Kim Bjerge
"""

import os

#%% MAIN
if __name__=='__main__':
    
    for shot in range(10):
        cmdLine = 'python TestFewShotNoveltyModel.py --model resnet50 --weights euMoths --dataset euMoths --learning True --shot %d --way 5 --query 6' % (shot+1)
        print(cmdLine)
        os.system(cmdLine)
    for shot in range(10):
        cmdLine = 'python TestFewShotNoveltyModel.py --model resnet50 --weights CUB --dataset CUB --learning True --shot %d --way 5 --query 6' % (shot+1)
        print(cmdLine)
        os.system(cmdLine)
    for shot in range(10):
        cmdLine = 'python TestFewShotNoveltyModel.py --model resnet50 --weights ImageNet --dataset miniImageNet --learning True --shot %d --way 5 --query 6' % (shot+1)
        print(cmdLine)
        os.system(cmdLine)