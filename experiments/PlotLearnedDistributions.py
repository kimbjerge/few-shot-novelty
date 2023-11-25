# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:56:01 2023

@author: Kim Bjerge
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
#import plotly.graph_objs as go

def funcExp(x, a, b, c):
    return b - a*np.exp(-c*x)

def funcLog(x, a, b, c):
    return a * np.log(b * x) + c

# Getting r squared value:
# https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
def calculate_r_squared(x, y, popt, func):
    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals**2) #  residulas sum of squares
    ss_tot = np.sum((y-np.mean(y))**2) #total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

#%% MAIN
if __name__=='__main__':
    
    
    #learned_distribution_resnet18_CUB = pd.read_csv("CUB_resnet18_novelty_learn_6way.txt")
    learned_distribution_resnet18_CUB = pd.read_csv("CUB_resnet18_novelty_learn_5way.txt")
    
    # Plot average and learned threshold vs. Few Shot
    ax = plt.gca()
    learned_distribution_resnet18_CUB.plot(kind='line',
                                           x='Shot',
                                           y='Threshold',
                                           color='red', ax=ax)
    
    learned_distribution_resnet18_CUB.plot(kind='line',
                                           x='Shot',
                                           y='Average',
                                           color='blue', ax=ax)
    
    # learned_distribution_resnet18_CUB.plot(kind='line',
    #                                        x='Shot',
    #                                        y='Std',
    #                                        color='green', ax=ax)
    plt.title('CUB dataset with ResNet18')
    plt.ylabel('Cosine similarity')
    plt.show()
    
    
    # Plot accuracy vs. Few Shot
    ax = plt.gca()
    learned_distribution_resnet18_CUB.plot(kind='scatter',
                                           x='Shot',
                                           y='Accuracy',
                                           color='red', ax=ax)
    
    plt.title('CUB dataset with ResNet18')
    plt.ylabel('Accuracy')
    plt.show()
    
    # Find non-liniear (logarithmic) relationship between threshold and few shot
    x = np.linspace(1,5,100)
    y = funcLog(x, 2.7, 1.3, 0.5)
    yn = y + 0.3*np.random.normal(size=len(x))
    
    popt, pcov = curve_fit(funcLog, x, y)
    print(popt)
    
    x = learned_distribution_resnet18_CUB['Shot'].to_numpy()
    y = learned_distribution_resnet18_CUB['Threshold'].to_numpy()
    popt, pcov = curve_fit(funcLog, x, y)
    print(popt)
    
    R2 = calculate_r_squared(x, y, popt, funcLog)
    
    print("R^2", R2)
    
    # fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', name= 'data'))
    # fig.add_trace(go.Scatter(
    #     x=x, y=func(x, *popt),
    #     name='Fitted Curve'
    # ))
    yn = funcLog(x, *popt)
    plt.plot(x, yn, color='red')
    plt.scatter(x, y)
    plt.xlabel('Shot')
    plt.ylabel('Threshold')
    textTitle = "Fitting log function y=a*ln(b*x)+c, R-square %.3f" % R2
    print("a %.5f b %.5f c %.5f" % (popt[0], popt[1], popt[2]))
    plt.title(textTitle)
    plt.show()
    
    # Find non-liniear (exponential) relationship between threshold and few shot
    x = np.linspace(1,5,100)
    y = funcExp(x, 0.239, 0.729, 0.601)
    yn = y + 0.3*np.random.normal(size=len(x))
    
    popt, pcov = curve_fit(funcExp, x, y)
    print(popt)
    
    x = learned_distribution_resnet18_CUB['Shot'].to_numpy()
    y = learned_distribution_resnet18_CUB['Threshold'].to_numpy()
    popt, pcov = curve_fit(funcExp, x, y)
    print(popt)
    
    R2 = calculate_r_squared(x, y, popt, funcExp)
    
    print("R^2", R2)
    
    # fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', name= 'data'))
    # fig.add_trace(go.Scatter(
    #     x=x, y=func(x, *popt),
    #     name='Fitted Curve'
    # ))
    yn = funcExp(x, *popt)
    plt.plot(x, yn, color='red')
    plt.scatter(x, y)
    plt.xlabel('n-shot')
    plt.ylabel('Threshold (th)')
    textTitle = "Fitting exponential function, R-square %.3f" % R2  # y=b-a*e(-c*x)
    print("a %.5f b %.5f c %.5f" % (popt[0], popt[1], popt[2]))
    plt.title(textTitle)
    plt.show()
    