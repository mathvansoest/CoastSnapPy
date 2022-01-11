# CoastSnap Python: Read the input image and retrieve metadata
"""
This script will import the specified image and deterime its resolution and other 
properties necesary for the CoastSnap Algorithm

Created by: Math van Soest
"""

import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

class CSim:
    
    def __init__(self, filename, path=os.getcwd()):
        self.path = path
        self.filename = filename
        self.data = cv.imread(os.path.join(path,filename))
        self.color = cv.cvtColor(self.data, cv.COLOR_BGR2RGB)
        self.gray = cv.cvtColor(self.data, cv.COLOR_BGR2GRAY)
        self.resolution = self.gray.shape
        self.NU = self.resolution[1]
        self.NV = self.resolution[0]
        self.c0U = self.NU/2
        self.c0V = self.NV/2

if __name__ == '__main__':
    path = 'C:\Coastal Citizen Science\CoastSnapPy'
    filename = '1616337600.Sun.Mar.21_15_40_00.CET.2021.egmond.snap.WouterStrating.jpg'
    
    im = CSim(path, filename)
    
    print(im.resolution)
    plt.imshow(im.color)