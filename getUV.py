# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:47:14 2022

@author: 4105664
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def getUV(path,im, GCPs):
    im = plt.imread(os.path.join(path,im))
    fig, axes = plt.subplots()
    axes.imshow(im)
    UV = np.zeros([2,len(GCPs)])

    for i,GCP in enumerate(GCPs):
        plt.title('Select %s' % (str(GCP)))
        fig.canvas.draw()
        fig.canvas.toolbar.zoom()
        while not plt.waitforbuttonpress(): pass
        fig.canvas.toolbar.zoom()
        axes.margins(0)
        A = plt.ginput(n=1, timeout=0, show_clicks=True)
        axes.plot(A[0][0], A[0][1],'go', markersize = 4)
        fig.canvas.draw()
        
        UV[0,i] = round(A[0][0])
        UV[1,i] = round(A[0][1])
    plt.close()
    
    return UV
    
if __name__ == "__main__":
    UV = getUV('C:\Github\CoastSnap\Target\egmond','target1.jpg',['Nautilus','P4','P5'])
    print(UV)