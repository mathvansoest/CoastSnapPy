# CoastSnap Python: function for image rectification using object detection
"""
This function will allow users CoastSnap images to be registered to an original image. 
Optionally: when there are to many unstable features within the picture frame, AI is used 
to detect stable objects. Everything else will be masked in order to perform accurate image 
registration. 
    
Created by: Math van Soest
"""

import cv2
import os
from imageai.Detection.Custom import CustomObjectDetection
import numpy as np

class CSdetection:
    def __init__(self, imPath, Objects, DetectionModels, objPath, ThresholdPercentage = 5):
        
        self.imPath = imPath
        self.Objects = Objects
        self.nObjects = len(Objects)
        self.DetectionModels = DetectionModels
        self.objPath = objPath
        self.ThresholdPercentage = ThresholdPercentage    
        
    def detector(self, imFileName):
        
        self.imFileName = imFileName  
        
        # initialize emtpy array for detection box pixel index
        self.points = np.zeros([self.nObjects,4])
        
        # Check if number of specified objects matches the number of specified detection models
        if np.size(self.Objects) and np.size(self.DetectionModels) == np.size(self.Objects):
            # Perform Detection for each of the specified objects
            for Object in self.Objects:
                objpath = os.path.join(os.path.abspath(self.objPath), Object)
            
                # Which object is being detected?
                print('detecting ' + Object + ' ...')
                # Load Custom Object Detection information
                detector = CustomObjectDetection()
                detector.setModelTypeAsYOLOv3()
                detector.setModelPath(os.path.join(objpath,"models",self.DetectionModels[self.Objects.index(Object)])) 
                detector.setJsonPath(os.path.join(objpath,"json\detection_config.json"))
                detector.loadModel()

                # Perform detection
                self.detections = detector.detectObjectsFromImage(input_image=os.path.join(self.imPath,self.imFileName),
                                                                  output_image_path=os.path.join(self.imPath,'detected_'+self.imFileName),
                                                                  minimum_percentage_probability = self.ThresholdPercentage)
                
                # Initialize emtpy array for storing the detection probabilities, this allows multiple instances of 
                # one object to be detected
                self.Prob = np.zeros(len(self.detections))
                
                # If multiple objects are detected check which one has the highest probability
                for j in range(len(self.detections)):
                    DetectDict = self.detections[j]
                    self.Prob[j]=DetectDict['percentage_probability']
                
                # Select only the detected object with the highest probability    
                detection = self.detections[np.argmax(self.Prob)]
                
                # 
                self.points[self.Objects.index(Object),:] = np.array(detection.get('box_points', 'Value'))
                
                print(Object + ' was detected with a probability of ' + "%.2f" % max(self.Prob) + '%')
                
            if len(self.points) != len(self.Objects):
                print('not all objects were found in the picture')
                
            return self.points
                
        else:
            print('No objects for detection were specified or the amount of specified objects does not correspond with the amount of specified models')

    def mask(self, addBoundary = True, boundaryPercentage = 50):
        
        self.imColor = cv2.imread(os.path.join(self.imPath,self.imFileName))
        self.imGray = cv2.cvtColor(self.imColor, cv2.COLOR_BGR2GRAY)
        self.imMaskMat = np.zeros(np.shape(self.imGray))
        
        for i in range(len(self.points)):

            x1=round(self.points[i,0])
            x2=round(self.points[i,2])
            y1=round(self.points[i,1])
            y2=round(self.points[i,3])
        
            if addBoundary == True:
            
                # Define size of object detection box
                xSize = x2 - x1
                ySize = y2 - y1
                # Adjust box size accordingly
                x1 = round(x1 - boundaryPercentage/100 * xSize)
                x2 = round(x2 + boundaryPercentage/100 * xSize)
                y1 = round(y1 - boundaryPercentage/100 * ySize)
                y2 = round(y2 + boundaryPercentage/100 * ySize)
                
                # Make sure added boundary is not outside of picture bounds
                if x1<0:x1=0
                if x2>np.size(self.imGray,1):x2=np.size(self.imGray,1)
                if y1<0:y1=0
                if y2>np.size(self.imGray,0):y2=np.size(self.imGray,0)
    
            # Create Mask file
            self.imMaskMat[y1:y2,x1:x2]=255
            
        self.imMaskMat = self.imMaskMat.astype(np.uint8)
            
        return self.imMaskMat
    
    def create_mask_target(self, targetDir, maskPath):
        
        ImList = os.listdir(self.imPath)
        Im = ImList[0]
        
        
        # Check if every image has a mask 
        if os.listdir(maskPath)[0] != 'target_mask.png': 
            
            #Perfrom object detection and retrieve pixel index of detection box
            imDetect = CSdetection(targetDir,self.Objects,self.DetectionModels,self.objPath,ThresholdPercentage=self.ThresholdPercentage)
            imDetect.detector(Im)
            mask = imDetect.mask()
            
            cv2.imwrite(os.path.join(maskPath,'target_mask.png'),mask)
                
        else:
            print('target_mask.png already exists')
        
    
    
    

