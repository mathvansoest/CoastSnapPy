# CoastSnap Python: function for image registration
"""
This function will allow users CoastSnap images to be registered to an original image. 
If object detection is for masked everything but the stable features in the image frame these
can be used as input for the function
    
Created by: Math van Soest
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from CSdetection import CSdetection


class CSregistration:
    def __init__(self,imReference,imRegister,imRegisterOriginal):
        self.imReference = imReference
        self.imRegister = imRegister
        self.imRegisterOriginal = imRegisterOriginal
            
        
    def register(self, nFeatures = 5000, matchPercent=0.15):    
        # Check if image is already in gray-scale
        if len(np.shape(self.imRegister)) == 3:
            self.imRegister = cv2.cvtColor(self.imRegister, cv2.COLOR_BGR2GRAY)
        
        if len(np.shape(self.imReference)) == 3:
            self.imReference = cv2.cvtColor(self.imReference, cv2.COLOR_BGR2GRAY)
            
        MAX_FEATURES = nFeatures
        GOOD_MATCH_PERCENT = matchPercent

        # Convert images to grayscale
        imReg = self.imRegister
        imRef = self.imReference
        imRegOr = self.imRegisterOriginal

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(imReg, None)
        keypoints2, descriptors2 = orb.detectAndCompute(imRef, None)
      
        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
      
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
      
        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
      
        # Draw top matches
        imMatches = cv2.drawMatches(imReg, keypoints1, imRef, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)
      
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
      
        for i, match in enumerate(matches):
          points1[i, :] = keypoints1[match.queryIdx].pt
          points2[i, :] = keypoints2[match.trainIdx].pt
      
        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
      
        # Use homography
        height, width = np.shape(imRef)
        imRegOr = cv2.warpPerspective(imRegOr, h, (width, height))
      
        return imRegOr, imMatches

# if __name__ == '__main__':

#   # Read reference image
#   refFilename = "form.jpg"
#   print("Reading reference image : ", refFilename)
#   imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

#   # Read image to be aligned
#   imFilename = "scanned-form.jpg"
#   print("Reading image to align : ", imFilename);
#   im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

#   print("Aligning images ...")
#   # Registered image will be resotred in imReg.
#   # The estimated homography will be stored in h.
#   imReg, h = alignImages(im, imReference)

#   # Write aligned image to disk.
#   outFilename = "aligned.jpg"
#   print("Saving aligned image : ", outFilename);
#   cv2.imwrite(outFilename, imReg)

#   # Print estimated homography
#   print("Estimated homography : \n",  h)
            
if __name__ == '__main__':
    
    imTest = '1616337600.Sun.Mar.21_15_40_00.CET.2021.egmond.snap.WouterStrating.jpg'
    
    Detect1 = CSdetection(imTest)
    Detect1.detector(Objects = ['strandtent','zilvermeeuw'], DetectionModels = ['detection_model-ex-016--loss-0008.891.h5','detection_model-ex-005--loss-0016.168.h5'], ThresholdPercentage=10)
    mask1 = Detect1.mask()
    
    imTest2 = '1610980860.Mon.Jan.18_15_41_00.CET.2021.egmond.snap.MargaVisser.jpg'
    
    Detect2 = CSdetection(imTest2)
    Detect2.detector(Objects = ['strandtent','zilvermeeuw'], DetectionModels = ['detection_model-ex-016--loss-0008.891.h5','detection_model-ex-005--loss-0016.168.h5'], ThresholdPercentage=10)
    mask2 = Detect2.mask()
    
    TestReg = CSregistration(mask1,mask2,Detect2.imRGB)
    
    #%%
    imRegistered = TestReg.register(nFeatures=5000,matchPercent=0.3)
    
    #%%
    fig1, ax = plt.subplots(2)
    ax[0].imshow(mask1)
    ax[1].imshow(mask2)
    fig2, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(imRegistered,cv2.COLOR_BGR2RGB))
    fig3, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(Detect1.imRGB,cv2.COLOR_BGR2RGB))
        

    
    

        
