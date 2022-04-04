# CoastSnap Python: This is the main script of the algorithm
"""
This script import all of the classes and function for CoastSnap Python to operate

Created by: Math van Soest
"""

import os
import sys
from CSreadDB import CSreadDB
from CSreadIm import CSim
from CSdetection import CSdetection
from CSrectification import CSrectification
from CSorganizer import CSorganizer
from CSmapSL import CSmapSL
from CSplotter import CSplotter
from CSregister2 import CSregister
import matplotlib.pyplot as plt
import numpy as np

def CoastSnapPy(sitename,new_im,outputPath=None): 
      
    # Do not show prints when specified
    # if output_print == False:
    #     sys.stdout = open(os.devnull, 'w')

    # Use oragnizer class to process new image
    organizer = CSorganizer(new_im,sitename,outputPath)
    organizer.check_filename()
    organizer.check_directories()
    organizer.gen_paths()
    organizer.process_new_image()
    
    # Retrieve new image file name from organizer class
    imname = organizer.NewImageName
    
    # Read CoastSnap xcel-database
    CSdb = CSreadDB(organizer.pathDB, sitename)
    
    # Check if each target image has defined UV points
    CSdb.checkUV(organizer.pathTarget)
    
    # Read the new image data
    im = CSim(imname, path=organizer.pathIm)
    
    # Detect the specified for detection with the corresponding models
    # Initialize detection class
    detect = CSdetection(imPath=organizer.pathIm,outPath=organizer.pathDetect,objPath=organizer.pathObjects,Objects = CSdb.ObjectNames, DetectionModels = CSdb.ObjectModels)
    # Perform object detection
    detect.detector(organizer.NewImageName)
    
    # Create mask where objects are located
    im.mask = detect.mask(addBoundary=True)

    # Check if all target images used for registration have their corresponding mask
    # if not, these are created
    detect.mask_target(organizer.pathTarget)
    
    #%% Perform registration of new image file
    # im.reg, best_match_tar = CSregister(im.color,im.mask,organizer.pathTarget)
    im.reg, best_match_tar = CSregister(im.color,
                                        im.mask,
                                        organizer.pathTarget,
                                        show_progress = False)
    
    # Get UV points from best match target image
    CSdb.UV = CSdb.getUV(best_match_tar)
            
    #%% Reference Image
    
    # Read the image data
    ref = CSim(best_match_tar, path=organizer.pathTarget)
    
    # fig1, ax1 = plt.subplots()
    # ax1.imshow(im.reg)
    # fig2, ax2 = plt.subplots()
    # ax2.imshow(ref.color)
    
    #%% Georectification
        
    rect = CSrectification(CSdb,im,CSdb.UV,registeredIm = True)
    
    # plt.figure(1)
    # fig2, axes = plt.subplots()
    # axes.imshow(im.reg)
    # axes.plot(CSdb.UV[0,:], CSdb.UV[1,:],'go', markersize = 3)
    # axes.scatter(rect.UV_pred[0,:], rect.UV_pred[1,:], s=80, facecolors='none', edgecolors='r')
    
    #%% Shoreline Mapping
   
    SL = CSmapSL(organizer.fileTrans,CSdb,im,rect)

    #%% Process the output files to their correct directory
    
    # Store registered image
    organizer.process_output_files(im.reg, '.reg')
    
    # Store rectified image
    organizer.process_output_files(rect.im, '.rect')
    
    # Store SL coordinates
    organizer.process_output_files(SL.xymat, '.SL')
    
    #%% User feedback images retrieved from plotter class
    
    # Use plotter class to make plot from mapped shoreline
    plotter = CSplotter()
    plotter.plot_rectSL_xyz(rect,SL,CSdb)
    
    # Define path and filename of plot to be returned to the user
    user_plot_file = os.path.join(organizer.pathPlot,organizer.NewImageName + '_plot.jpg')
    
    # Save plot
    plt.savefig(user_plot_file,orientation='portrait',dpi=400)
    
    # Switch on prints when switched off before
    # if output_print == False:
    #     sys.stdout = sys.__stdout__

    return user_plot_file

#%% Run function

user_plot = CoastSnapPy('egmond','test2.jpg')
