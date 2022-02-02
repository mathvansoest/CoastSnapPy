# CoastSnap Python: This is class will take care of plotting user feedback
"""
This script is a class that allows to plot and do the 
lay-out of the user feedback images. Mainly a rectified image with the 
corresponding location of the shoreline will be provided. In addition 
graphs of beach width trend are provided.

Created by: Math van Soest
"""

import matplotlib.pyplot as plt

class CSplotter():
    
    def __init__ (self):
        self = self
    
    def plot_rectSL_xyz(self,rect,SL,CSinput):
        
        plt.figure()
        plt.plot(SL.x[0,:],SL.y[0,:], color = 'r')
        plt.imshow(rect.im, extent = (min(CSinput.x), max(CSinput.x), max(CSinput.y), min(CSinput.y)))
        plt.xlabel("M")
        plt.ylabel("N")
        plt.gca().invert_yaxis()
        plt.title("Corrected Shoreline (Python)")
    
    #TODO def plot_rectSL_UTM(self,rect,SL,CSinput):