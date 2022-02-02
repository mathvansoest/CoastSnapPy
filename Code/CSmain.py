# CoastSnap Python: This is the main script of the algorithm
"""
This script import all of the classes and function for CoastSnap Python to operate

Created by: Math van Soest
"""

from CSreadDB import CSinput
from CSreadIm import CSim
from CSdetection import CSdetection
from CSregistration2 import CSregistration
from CSrectification import CSrectification
from CSorganizer import CSorganizer
from CSmapSL import CSmapSL
from CSplotter import CSplotter
from CSregister import register_img
import cv2 as cv
import matplotlib.pyplot as plt
import angles2R
import numpy as np


if __name__ == '__main__':

    #%% Set up with use of CSorganizer
    
    sitename = 'egmond'
    new_im = 'test2.jpg'
    
    organizer = CSorganizer(new_im,sitename)
    organizer.check_time()
    organizer.check_directories()
    organizer.gen_paths()
    organizer.process_new_image()
    
    imname = organizer.NewImageName
    
    refname = 'target_image23.jpg'
    
    # Define the path, names of objects to be detected and their corresponding detection model names
    objects = ['strandtent',
               'zilvermeeuw']
    detection = ['detection_model-ex-016--loss-0008.891.h5',
                 'detection_model-ex-005--loss-0016.168.h5']
    # Define the percentage threshold for object detection
    detectionThreshold = 5 #[%]
    
    # Number of features used for image registration
    
    #%% Read Database
    CSinput = CSinput(organizer.pathDB, sitename)
    
    #%% Input Image
    
    # Read the image data
    im = CSim(imname, path=organizer.pathIm)
    # Detect the specified for detection with the corresponding models
    imDetect = CSdetection(imPath=organizer.pathIm,objPath='C:\Coastal Citizen Science\CoastSnapPy\CoastSnap\Objects\egmond',Objects = objects, DetectionModels = detection)
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
    ref = CSim(refname, path=organizer.pathTargetIm)
    
    
    fig1, ax1 = plt.subplots()
    ax1.imshow(im.reg)
    fig2, ax2 = plt.subplots()
    ax2.imshow(ref.color)
    
    #%% Object Detection for UV retrieval

    #%% Georectification
    
    UV = np.array([[3175.16433347455, 2411.05506640388, 2934.31631638085],
                   [1208.15196942975, 1426.95972957084, 1245.35361552028]])
    
    # UV = np.array([[3318.53598243924, 2546.07875045151, 3072.77294327758],
    #                [1205.66917219563, 1419.20272388791, 1238.25280647238]])
        
    rect = CSrectification(CSinput,im,UV,registeredIm = True)
    
    plt.figure(1)
    fig2, axes = plt.subplots()
    axes.imshow(im.reg)
    axes.plot(UV[0,:], UV[1,:],'go', markersize = 3)
    axes.scatter(rect.UV_pred[0,:], rect.UV_pred[1,:], s=80, facecolors='none', edgecolors='r')
    
    #%% Shoreline Mapping
   
    SL = CSmapSL(organizer.fileTrans,CSinput,im,rect)

    #%% Process the output files to their correct directory
    
    # Store registered image
    organizer.process_output_files(im.reg, 'reg')
    
    # Store rectified image
    organizer.process_output_files(rect.im, 'rect')
    
    # Store SL coordinates
    organizer.process_output_files(SL.xymat, 'SL')
    
    #%% User feedback images retrieved from plotter class
    
    plotter = CSplotter()
    plotter.plot_rectSL_xyz(rect,SL,CSinput)