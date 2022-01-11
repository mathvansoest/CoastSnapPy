# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:57:47 2021

@author: 4105664
"""
import numpy as np

def angles2R(a, t, s):
    R = np.zeros((3, 3))
    
    R[0,0] = np.cos(a) * np.cos(s) + np.sin(a) * np.cos(t) * np.sin(s)
    R[0,1] = -np.cos(s) * np.sin(a) + np.sin(s) * np.cos(t) * np.cos(a)
    R[0,2] = np.sin(s) * np.sin(t)
    R[1,0] = -np.sin(s) * np.cos(a) + np.cos(s) * np.cos(t) * np.sin(a)
    R[1,1] = np.sin(s) * np.sin(a) + np.cos(s) * np.cos(t) * np.cos(a)
    R[1,2] = np.cos(s) * np.sin(t)
    R[2,0] = np.sin(t) * np.sin(a)
    R[2,1] = np.sin(t) * np.cos(a)
    R[2,2] = -np.cos(t)
    
    return R