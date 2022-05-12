# CoastSnap Python: This is class will take care of plotting user feedback
"""
This script is a class that allows to plot and do the 
lay-out of the user feedback images. Mainly a rectified image with the 
corresponding location of the shoreline will be provided. In addition 
graphs of beach width trend are provided.
Created by: Math van Soest
"""
import glob
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

class plotter():
    
    def __init__ (self,show_plots=True):
        self = self
        self.show_plots = show_plots
    
    def rectSL_xyz(self,rect,SL,db):
        
        if self.show_plots == False:
            plt.ioff()
            
        plt.figure()
        plt.plot(SL.x[0,:],SL.y[0,:], color = 'r')
        plt.imshow(rect.im, extent = (min(db.x), max(db.x), max(db.y), min(db.y)))
        plt.xlabel("Noord [m]")
        plt.ylabel("Oost [m]")
        plt.gca().invert_yaxis()
        plt.title("Waterlijn")
        plt.ion()

    def ref(self,ref):
        
        if self.show_plots == False:
            plt.ioff()
        
        plt.figure()
        plt.imshow(ref.color)
        plt.title("Best match target image")
        plt.ion()
        
    def reg(self,im,rect,db):
        
        if self.show_plots == False:
            plt.ioff()
        
        plt.figure()
        plt.imshow(im.reg)
        plt.plot(db.UV[0,:], db.UV[1,:],'go', markersize = 3)
        plt.scatter(rect.UV_pred[0,:], rect.UV_pred[1,:], s=80, facecolors='none', edgecolors='r')
        plt.title("Registered image with GCP's")
        plt.ion()
        
    def rect(self,rect,db,transect):
        
        if self.show_plots == False:
            plt.ioff()
        
        trans = scipy.io.loadmat(transect)['SLtransects']
        
        x = trans['x'][0][0].astype(float)
        y = trans['y'][0][0].astype(float)
        plt.figure()
        plt.plot(x,y)
        plt.imshow(rect.im, extent = (min(db.x), max(db.x), max(db.y), min(db.y)))
        plt.xlabel("M")
        plt.ylabel("N")
        plt.gca().invert_yaxis()      
        plt.ion()
        
        x = (x - db.xlim[0]) * 1/db.dxdy
        y = (y - db.ylim[0]) * 1/db.dxdy
        
        plt.figure()
        plt.plot(x,y)
        plt.imshow(rect.im)
        plt.xlabel("M")
        plt.ylabel("N")
        plt.gca().invert_yaxis()      
        plt.ion()
        
    def trend(self,pathSL,transect,trend_points=20):
        
        if self.show_plots == False:
            plt.ioff()
    
        # List all files from Shoreline directory
        files = glob.glob(os.path.join(pathSL,'*.mat'))
        
        # Initialize empty list
        recent = []
        BeachWidth = np.array([])
        
        # Iterate for as many trend points are needed for the plot
        for i in range(trend_points):
            
            # Determine the newest file
            newest = max(files,key=os.path.getctime)

            # Append that file to a list of necessary SL-files for the plot 
            recent.append(newest)
            # Remove the latest so that the next most recent is selected in the following iteration
            files.remove(newest)
            
            # Read the shoreline data form the most recent file and for the specified transect only x-coordinate
            SL_recent = np.array(scipy.io.loadmat(newest)['xyz'][transect][0])
            
            # Append the x coordinate form the SL coordinate as beachwidth to be plotted
            BeachWidth = np.append(BeachWidth,abs(SL_recent))
        
        # Set x for each of the latest CoastSnap    
        x = range(trend_points)  
        plt.figure()
        plt.scatter(x,BeachWidth)
        plt.ion()
        # Determine the trendline
        z = np.polyfit(x, BeachWidth, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"r--")
        plt.xlabel('Snaps')
        plt.ylabel('Beach Width [m]')
        plt.title('Trend')
        
if __name__ == '__main__':
    path = r'C:\Github\CoastSnap\Shorelines\egmond\2022'
    
    plot = CSplotter()
    plot.trend(path,50)
    
    #TODO def plot_rectSL_UTM(self,rect,SL,CSinput):