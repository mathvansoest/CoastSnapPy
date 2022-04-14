# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:38:52 2022
@author: 4105664
"""

import cv2
import os
import glob
import numpy as np
from tqdm import tqdm


#%% Load all reference images

def register(newIm,
               imMask,
               targetDir,
               imPath = os.getcwd(),
               nfeatures=1000,
               score_method = 'distance',
               max_distance=False,
               max_distance_value=50,
               same_region=False,
               same_region_threshold=0.2,
               ransac_threshold=10,
               homography_confidence=0.9,
               imMatches=False,
               show_progress=True):

    # Define file extension, target images should be .jpg's, mask files 
    # should be .png's
    jpg_extension = os.path.join(targetDir, r"*.jpg")
    png_extension = os.path.join(targetDir, r"*.png")
    
    # List all target and mask files
    tar_list = [os.path.basename(x) for x in glob.glob(jpg_extension)]
    mask_list = [os.path.basename(x) for x in glob.glob(png_extension)]
 
    #Initialize ORB features
    orb = cv2.ORB_create(nfeatures)
    # Draw keypoints on the Raw new image
    keypointsR, descriptorsR = orb.detectAndCompute(newIm,mask=imMask)
    # Determine corner values for new image
    cornersR = np.array([[0,0],                             
                         [0,newIm.shape[1]],                 
                         [newIm.shape[0],0],                 
                         [newIm.shape[0],newIm.shape[1]],
                         [newIm.shape[0]/2,newIm.shape[1]/2]])
    
    
    # intialize empty arrays
    h_all = np.empty([3,3,len(tar_list)])
    h_det_all = np.array([])
    d_all = np.array([])
    
    # Total iterations
    total_i = len(tar_list)
    
    # Initialize progress bar
    with tqdm(total=total_i, leave = True, disable=not show_progress) as pbar:
    
        # Iterate over all target images
        for i, (tar,mask) in enumerate(zip(tar_list,mask_list)):
            
            # Start progress bar
            pbar.set_description("Attempting registration with %s" % tar)
            
            # Load Target image in grayscale
            tar_gray = cv2.cvtColor(cv2.imread(os.path.join(targetDir,tar)), cv2.COLOR_BGR2GRAY)
            
            # Load Corresponding Target Mask
            tar_mask = cv2.imread(os.path.join(targetDir,mask), cv2.COLOR_BGR2GRAY)
            
            # Draw Keypoints
            keypointsT, descriptorsT = orb.detectAndCompute(tar_gray,mask=tar_mask)
            
            # Initialize Brute Force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            if max_distance == True:
                # Delete all matches that surpass max distance
                all_matches = bf.match(descriptorsR, descriptorsT)
                matches = []
                for m in all_matches:
                    if m.distance < max_distance_value:
                        matches.append(m)
            else:
                matches = bf.match(descriptorsR, descriptorsT)
                
            # Matches should be in the same region of the image. Use the threshold to set
            # how far matches can be away from each other (spatially) before discarding
            # as a bad match
            if same_region == True:
            
                max_dim = max(tar_gray.shape)
                threshold = same_region_threshold
                
                # See how far keypoints are spatially. Iterate backwards since we are
                # removing items from the list
                for m in matches[::-1]:
                
                    raw_x, raw_y = keypointsR[m.queryIdx].pt
                    target_x, target_y = keypointsT[m.trainIdx].pt
                
                    spatial_distance = np.sqrt(
                        (raw_x - target_x) ** 2 + (raw_y - target_y) ** 2
                    )
                
                    if spatial_distance > max_dim * threshold:
                        matches.remove(m)
            
    
            # TODO only do this for best match
            if imMatches == True:
                match_im = cv2.drawMatches(newIm, keypointsR, tar_gray, keypointsT, matches,None)
                return match_im
            
            # Get key points from raw image and target image
            raw_pts = np.float32([keypointsR[m.queryIdx].pt for m in matches])
            target_pts = np.float32([keypointsT[m.trainIdx].pt for m in matches])
            
            # Compute homography and store homography matrix in h
            h,_ = cv2.findHomography(raw_pts, target_pts, cv2.RANSAC, ransacReprojThreshold=ransac_threshold, confidence=homography_confidence)
            
            # Store all homography matrices
            h_all[:,:,i] = h
            
            # Use homography matrix to check euclidean distance between GCPs pixel
            # location of the target image with the registered image
            if score_method == 'distance':
                
                # Determine corner values for new image
                cornersT = np.array([[0,0],                                   # Top left                              
                                     [0,tar_gray.shape[1]],                   # Top right
                                     [tar_gray.shape[0],0],                   # Bottom left
                                     [tar_gray.shape[0],tar_gray.shape[1]],   # Bottom right
                                     [tar_gray.shape[0]/2,tar_gray.shape[1]/2]])# Center
                
                # Project corners of raw image
                projected_cornersR = cv2.perspectiveTransform(cornersR.reshape(cornersT.shape[0],1,2).astype(float),h)
                projected_cornersR = projected_cornersR.reshape([cornersT.shape[0],2])
                
                # Calculate Euclidean distance between corners of the raw and target image
                dist = sum(np.sqrt((projected_cornersR[:,0]-cornersT[:,0])**2+(projected_cornersR[:,1]-cornersT[:,1])**2))
                
                # Store all distance values
                d_all = np.append(d_all,dist)

            # Use the determinant of the homography matrix to score the registration
            # of a target image with a new raw image
            elif score_method == 'h_det':
                
                # # Calculate the determinant of the top left 2x2 cells of the homography matrix 
                # # to check it's stability 
                h_det = np.linalg.det(h)
                
                # Store all h_det values
                h_det_all = np.append(h_det_all,np.array(h_det))
        
            # Update progress bar after each iteration
            pbar.update(1)
     
    # Check the scoring method        
    if score_method == 'distance':
        
        # find index of the lowest summed euclidean distance between target and raw corner values
        best_d_index = np.argmin(d_all)
        # find corresponding h_det value
        best_d_value = d_all[best_d_index]
        # get the homography matrix for the best match
        best_h = h_all[:,:,best_d_index]
        # get the target image with which the best match was found    
        best_match_tar = tar_list[best_d_index]
        
        # Tell which target image resulted in h_det closest to 1
        print('Best match with %s with a euclidean distance score of %s' % (best_match_tar,str(round(best_d_value,3))))
    
    # Check the scoring method
    elif score_method == 'h_det':
        
        # find index of the determinant of the homography matrix closest to 1
        best_h_index = np.argmin(abs(1-h_det_all))
        # find corresponding h_det value
        best_h_det_value = h_det_all[best_h_index]
        # get the homography matrix for the best match
        best_h = h_all[:,:,best_h_index]
        # get the target image with which the best match was found    
        best_match_tar = tar_list[best_h_index]  
        
        # Tell which target image resulted in h_det closest to 1
        print('Best match with %s with a homography determinant score of %s' % (best_match_tar,str(round(best_h_det_value,3))))
    
    # load the best match target image
    im_tar = cv2.imread(os.path.join(targetDir,best_match_tar))
        
    # Perform registration
    reg_im = cv2.warpPerspective(newIm,best_h,(im_tar.shape[1],im_tar.shape[0]))    
    
    
    
    return reg_im, best_match_tar