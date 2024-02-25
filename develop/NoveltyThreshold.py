# -*- coding: utf-8 -*-
"""
Modified on Sun Feb 25 11:34:31 2024

@author: Kim Bjerge
"""
import math 

# function for finding roots in second order equation
def equationroots(a, b, c): 
 
    # calculating discriminant using formula
    dis = b * b - 4 * a * c 
    sqrt_val = math.sqrt(abs(dis)) 
     
    # checking condition for discriminant
    if dis > 0: 
        print("real and different roots") 
        x1 = (-b + sqrt_val)/(2 * a)
        x2 = (-b - sqrt_val)/(2 * a)
        print(x1, x2) 
     
    elif dis == 0: 
        print("real and same roots")
        x1 = -b / (2 * a)
        x2 = 0
        print(x1) 
     
    # when discriminant is less than 0
    else:
        print("Complex Roots") 
        #print(- b / (2 * a), + i, sqrt_val) 
        #print(- b / (2 * a), - i, sqrt_val) 

    x = x1
    if x2 > 0 and x2 < 1: # Use the solution between 0-1
        x = x2
    return x, x2    

# Bayes classification for two classes 
# using the decision function d(x) = p(x|w)*P(w)
# k is the distribution of the known classes
# o is the distribution of the outliers 
def BayesTwoClassThreshold(var_k, mean_k, var_o, mean_o, k_way, m_outlier):
    
    std_k = math.sqrt(var_k)
    std_o = math.sqrt(var_o)
    a = (var_k - var_o)
    b = -2*(mean_o*var_k - mean_k*var_o)
    P = (math.sqrt(std_k)/math.sqrt(std_o))*(m_outlier/k_way)
    c = var_k*math.pow(mean_o,2) - var_o*math.pow(mean_k,2) - 2*var_o*var_k*math.log(P)  
    
    return equationroots(a, b, c)

def StdTimesTwoThredshold(mu_k, sigma_k):
    
    return mu_k - 2*sigma_k


