# CoastSnap Python: This is the main script of the algorithm
"""
This script import all of the classes and function for CoastSnap Python to operate

Created by: Math van Soest
"""

from CSreadDB import CSreadDB
from CSreadIm import CSim
from CSdetection import CSdetection
from CSrectification import CSrectification
from CSorganizer import CSorganizer
from CSmapSL import CSmapSL
from CSplotter import CSplotter
from CSregister import register_img
import matplotlib.pyplot as plt
import numpy as np

def CoastSnapPy(sitename,new_im): 

    #%% Set up with use of CSorganizer
        
    # Use oragnizer class to process new image
    organizer = CSorganizer(new_im,sitename)
    organizer.check_time()
    organizer.check_directories()
    organizer.gen_paths()
    organizer.process_new_image()
    
    # Retrieve new image file name from organizer class
    imname = organizer.NewImageName

    #%% Read Database
    
    CSdb = CSreadDB(organizer.pathDB, sitename)
    
    #%% Input Image
    
    # Read the image data
    im = CSim(imname, path=organizer.pathIm)
    # Detect the specified for detection with the corresponding models
    imDetect = CSdetection(imPath=organizer.pathIm,objPath='C:\Coastal Citizen Science\CoastSnapPy\CoastSnap\Objects\egmond',Objects = CSdb.ObjectNames, DetectionModels = CSdb.ObjectModels)
    imDetect.detector(organizer.NewImageName)
    # Mask everything but the detected stable features
    im.mask = imDetect.mask(addBoundary=False)
    
    im.reg = register_img(organizer.NewImageName,
                          imagePath = organizer.pathIm,
                          targetPath = organizer.pathTargetIm,
                          mask = im.mask,
                          warp = 'perspective')
        
    #%% Reference Image
    
    # Read the image data
    ref = CSim(CSdb.RefImage, path=organizer.pathTargetIm)
    
    fig1, ax1 = plt.subplots()
    ax1.imshow(im.reg)
    fig2, ax2 = plt.subplots()
    ax2.imshow(ref.color)
    
    #%% Georectification
        
    rect = CSrectification(CSdb,im,CSdb.UV,registeredIm = True)
    
    plt.figure(1)
    fig2, axes = plt.subplots()
    axes.imshow(im.reg)
    axes.plot(CSdb.UV[0,:], CSdb.UV[1,:],'go', markersize = 3)
    axes.scatter(rect.UV_pred[0,:], rect.UV_pred[1,:], s=80, facecolors='none', edgecolors='r')
    
    #%% Shoreline Mapping
   
    SL = CSmapSL(organizer.fileTrans,CSdb,im,rect)

    #%% Process the output files to their correct directory
    
    # Store registered image
    organizer.process_output_files(im.reg, 'reg')
    
    # Store rectified image
    organizer.process_output_files(rect.im, 'rect')
    
    # Store SL coordinates
    organizer.process_output_files(SL.xymat, 'SL')
    
    #%% User feedback images retrieved from plotter class
    
    plotter = CSplotter()
    plotter.plot_rectSL_xyz(rect,SL,CSdb)

#%% Run function

CoastSnapPy('egmond','test1.jpg')
