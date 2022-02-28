# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:47:14 2022

@author: 4105664
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def getUV(path,im, GCPs):
    
    # Load image
    im = plt.imread(os.path.join(path,im))
    # Initialize and plot figure window
    fig, axes = plt.subplots()
    axes.imshow(im)
    # Initialize empty array to store UV
    UV = np.zeros([2,len(GCPs)])

    # Iterate over all of the needed GCP points
    for i,GCP in enumerate(GCPs):
        # Vary title depending on GCP
        plt.title('Select %s' % (str(GCP)))
        # Draw figuren and wait for button press to deselect zoom function
        fig.canvas.draw()
        fig.canvas.toolbar.zoom()
        while not plt.waitforbuttonpress(): pass
        fig.canvas.toolbar.zoom()
        # Reset figure zoom
        axes.margins(0)
        # Retrieve clicked pixel coordinates
        A = plt.ginput(n=1, timeout=0, show_clicks=True)
        axes.plot(A[0][0], A[0][1],'go', markersize = 4)
        # Redraw figure including the scattered pixel values
        fig.canvas.draw()
        
        # Store results in UV
        UV[0,i] = round(A[0][0])
        UV[1,i] = round(A[0][1])
    
    plt.close()
    
    return np.array(UV)