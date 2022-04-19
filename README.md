# CoastSnapPy - CoastSnap Python
Contact: Math van Soest (mathvansoest@gmail.com) 
Created by: Utrecht University

## About
This is an automated version of the original CoastSnap code as written in Matlab. MATLAB based code for registration and shoreline extraction can be found on Github.

It was developped in conjuction with the coastsnap.nl webapplication. It allows users to instantly share and process a photo taken from a CoastSnap station with our servers from Utrecht University. Automation of shoreline extraction is based on image registration and object detection.

## How to run
This version of CoastSnap was developed using the Spyder IDE. Spyder is a free and open source scientific environment written in Python, for Python, and designed by and for scientists, engineers and data analysts. This works in conjunction with Anaconda. In the CoastSnap folder there an environment.yml file can be found. This has to be used in order to install the necessary dependencies for the CoastSnapPy code.

--step1-- 
Download the Anaconda platform: https://www.anaconda.com/products/distribution. 

--step2-- 
Open Spyder through the Anaconda navigator. 

--step3-- 
Use the "cd" command to acces the CoastSnap directory in the Console. 

--step4--
Initialize the conda evironment running the following command in the Spyder console: "conda env -f create environment.yml" 

--step5-- 
Set the python interpreter to run within the newly created environment. In Spyder go to  "preferences ". Then "Python Interpreter", following select "Use the following python interpreter", then select "~coastsnappy/bin/python"  

--step6--
Download the object detection used for our NL CoastSnap Location Egmond from: ...
Paste ....h5 in "CoastSnapPy/Objects/egmond/strandtent/models and paste ....h5 in "CoastSnapPy/Objects/egmond/zilvermeeuw/models.

--step7--
Run CSmain.py with the test images form our CoastSnap site Egmond, the Netherlands to see if it is working!

## How does CoastSnapPy work
CoastSnapPy is a reinterpretation of CoastSnap. This an algorithm developed by the UNSW and its workings have been elaborated on in ... and the original code can be downloaded from: https://github.com/Coastal-Imaging-Research-Network/CoastSnap-Toolbox.

In the original code a user manually needs to identify the location of Ground Control Points in an image. CoastSnapPy aims at resolving this manual step by identifying these GCP's automatically. This is achieved using image registration/alignment. When a new image is registered perfectly to an image of which the location of the GCP's is known the location of the GCP's is identified in the new image. Based on the real world position of the GCP's an image can be georectified and the location of the shoreline on the image can be identified. 

## How to intialize a new CoastSnap location
### Settting up CoastSnapPyDB
To start you need to copy the 'egmond' sheet from the 'Database/CoastSnapPy.xlsx' file. Make sure that all cells remain in the same location (e.g. "Station Data" in A1). Then fill in the Station Data, Rectification Settings, Tide Data, Shoreline Mapping Settings, Ground Control Points, GCP Rectification Combo. Object Detection and GCP's target image can be left blank for now. 

### Active vs. Passive
In your CoastSnapPy.xlsx you can define whether a site is active or not. When you are retrieving images from a CoastSnap location where no database of images has been collected yet, you are not yet able to make use of the automated image processing. But you can use the data structure to organise the new incoming images. If this is the case you define cell A2 as 'INACTIVE'. When you have trained the object detection and have sufficient target images you can change this to 'ACTIVE' when the main function is then ran, the code will try derive the shoreline position from your CoastSnaps. 

### Training object detection
Next you need to find stable objects in your image frame and train the AI object detection to find and isolate the stable features. This process is clearly explained on: https://medium.com/deepquestai/train-object-detection-ai-with-6-lines-of-code-6d087063f6ff. Note: in order to have reliable training data you need a database of images already taken from your new CoastSnap location. For egmond we used 100 training images and 23 validating images. This works very reliably. It can probably be done with less. 

### Adding detection models
Once you have trained and evaluated the detection models you have to select the detection model with the mAP closest to 1. Copy the created folder for your custom object to "CoastSnapPy/Objects/your_object". Define the name of the object in the CoastSnapPyDB.xlsx next to Object Names. When using multiple add these (e.g. "restaurant,stairs,rock"). Next to models define the detection model to be used for each object you want to detect (e.g. "detection_model-ex-05--loss-5.26.h5"). When using multiple objects, make sure you define them in the same order as the objects specified in te cell above. 

### Adding target images
In order to achieve the most accurate image registration, multiple target images are used and tested for accurate registration to the new image. For CoastSnap Egmond we use 30 images and are able to retrieve stable results with that amount. Note: the more stable your image frame the less images you need. Also, the more images added the longer the code will need to run. Make sure that the collection of target images cover a wide variety of conditions. 

When adding the images the code will check two things: if every target image has a corresponding target..._mask.png and if de GCP's pixel location is defined in the CoastSnapPyDB.xlsx. If the target image is non existent, this will be generated using the objects as defined in the CoastSnapPyDB.xlsx. If the pixel values are non existent the user will be present a GUI where the pixel location can be defined by clicking on them, similar to the original Matlab code. The image without the GCP pixel location is presenten. By dragging the mouse you can zoom in. Then after pressing any button you can click on the location of the GCP. The pixel coordinates are then copied onto the clipboard and can be pasted in the CoastSnapDB.xlsx. 

## Options
### main
Most optional functionality of the code is determined in the main.py file. 

Optional input: 	  output_path = None/path to desired location
		  show_progress = True/False
	
ouput_path: if a path is defined for this argument all output that is usually stored in the 	images folder is then stored a folder called 'CSoutput' at the specified location. This allows you to decouple the location of the CoastSnapPy output from the working directory of the code. 

show_progress: a progress bar is printed for the image registration module. When this is set to false the progress bar is not printed.

### register
Optional input:     score_method =  h_det/distance
                    max_distance = True/False
                    max_distance_value = x amount of pixels
                    same_region = True/False
                    same_region_values = 0-1 part of image
                    ransac_threshold = x amount of pixels
                    homography_confidence = 0-1 confidence interval
                    imMatches = True/False
                    show_progress = True/False
                        
score_method: allows you to change the way in which the best target image is decided. When usig 'h_det' the determinant of the homography matrix is calculated. When this is close to 1 it is assumed that the matrix is stable and represents an accurate image transformation. When using 'distance' various pixel points such as the corners and middle of the image are transposed and checked for their 2D distance projection. When this distance is smallest it is assumed that the image is registration is stable. 
    
max_distance: only uses keypoints from the cv2.BFmatcher that fall within a certain distance of one another. This distance is defined by an amount of pixels in max_distance_value. Best to leave off.
    
same_region: only use keypoints that fall within a certain part of the image in order to reduce spread of the keypoints used for image registration best to leave off.
    
ransac_threshold: used in cv2.findHomography. Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC method only).
    
homography_confidence: confidence threshold for using a keypoint match
    
imMatches: print image the shows the matched keypoints
    
show_progress: show progress bar when iterating over eacht of the target images specified in the target directory. 

