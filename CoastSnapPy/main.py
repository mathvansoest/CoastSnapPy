# CoastSnap Python: This is the main script of the algorithm
"""
This script import all of the classes and function for CoastSnap Python to operate
Created by: Math van Soest
"""

import os
import matplotlib.pyplot as plt

from CoastSnapPy import (readDB, 
                         readIm,
                         detection,
                         rectification,
                         organizer,
                         mapSL,
                         plotter,
                         register)

def CoastSnapPy(sitename,new_im,outputPath=None,show_plots=False): 

    # Use oragnizer class to process new image
    organize = organizer(new_im,sitename,outputPath)
    organize.check_filename()
    organize.check_directories()
    organize.gen_paths()
    organize.process_new_image()
    
    # Retrieve new image file name from organizer class
    imname = organize.NewImageName
    
    # Read CoastSnap xcel-database
    db = readDB(organize.pathDB, sitename)
    
    # Check if each target image has defined UV points
    db.checkUV(organize.pathTarget)
    
    # Read the new image data
    im = readIm(imname, path=organize.pathIm)
    
    # Detect the specified for detection with the corresponding models
    # Initialize detection class
    detect = detection(imPath=organize.pathIm,
                       outPath=organize.pathDetect,
                       objPath=organize.pathObjects,
                       Objects = db.ObjectNames, 
                       DetectionModels = db.ObjectModels)
    # Perform object detection
    detect.detector(organize.NewImageName)
    
    # Create mask where objects are located
    im.mask = detect.mask(addBoundary=True)

    # Check if all target images used for registration have their corresponding mask
    # if not, these are created
    detect.mask_target(organize.pathTarget)
    
    # Perform registration of new image file
    im.reg, best_match_tar = register(im.color,
                                      im.mask,
                                      organize.pathTarget,
                                      show_progress = True)
    
    # Read the image data
    ref = readIm(best_match_tar, path=organize.pathTarget)
    
    # Get UV points from best match target image
    db.UV = db.getUV(best_match_tar)
    
    # Georectification   
    rect = rectification(db,im,db.UV,registeredIm = True)
         
    # Shoreline Mapping
    SL = mapSL(organize.fileTrans,db,im,rect)

    # Store registered image
    organize.process_output_files(im.reg, '.reg')
    
    # Store rectified image
    organize.process_output_files(rect.im, '.rect')
    
    # Store SL coordinates
    organize.process_output_files(SL.xymat, '.SL')
    
    # Use plotter class to make plot from mapped shoreline
    plot = plotter(show_plots=show_plots)
    plot.ref(ref)
    plot.reg(im,rect,db)
    plot.rectSL_xyz(rect,SL,db)  
    plot.trend(organize.pathSL,50)
    
    # Define path and filename of plot to be returned to the user
    user_plot_file = os.path.join(organize.pathPlot,organize.NewImageName + '_plot.jpg')
    
    # Save plot
    plt.savefig(user_plot_file,orientation='portrait',dpi=400)

    return user_plot_file

#%% Run function
if __name__ == "__main__":
    user_plot = CoastSnapPy('egmond','test1.jpg',show_plots=True)
