# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:38:52 2022

@author: 4105664
"""

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from CSreadIm import CSim
from CSorganizer import CSorganizer
from CSreadDB import CSreadDB
from tqdm import tqdm
from CSdetection import CSdetection

#%% Load all reference images

def CSregister(newIm,
               imMask,
               targetDir,
               imPath = os.getcwd(),
               nfeatures=10000,
               max_distance=False,
               max_distance_value=300,
               same_region=False,
               same_region_threshold=0.2,
               ransac_threshold=10,
               homography_confidence=0.95,
               imMatches=False):

    # Define file extension, target images should be .jpg's, mask files 
    # should be .png's
    jpg_extension = targetDir + r"\*.jpg"
    png_extension = targetDir + r"\*.png"
    
    # List all target and mask files
    tar_list = [os.path.basename(x) for x in glob.glob(jpg_extension)]
    mask_list = [os.path.basename(x) for x in glob.glob(png_extension)]
    
    # intialize empty arrays
    h_all = np.empty([3,3,len(tar_list)])
    h_det_all = np.array([])
    
    # Total iterations
    total_i = len(tar_list)
    # Initialize progress bar
    with tqdm(total=total_i, leave = True) as pbar:
    
        # Iterate over all target images
        for i, (tar,mask) in enumerate(zip(tar_list,mask_list)):
            
            # Start progress bar
            pbar.set_description("Attempting registration with %s" % tar)
            
            # Load Target image in grayscale
            tar_gray = cv2.cvtColor(cv2.imread(os.path.join(targetDir,tar)), cv2.COLOR_BGR2GRAY)
            
            # Load Corresponding Target Mask
            tar_mask = cv2.imread(os.path.join(targetDir,mask), cv2.COLOR_BGR2GRAY)
            
            #Initialize ORB features
            orb = cv2.ORB_create(nfeatures)
            
            # Draw Keypoints
            keypointsR, descriptorsR = orb.detectAndCompute(newIm,mask=imMask)
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
            
            # Get key points from raw image and target image
            raw_pts = np.float32([keypointsR[m.queryIdx].pt for m in matches])
            target_pts = np.float32([keypointsT[m.trainIdx].pt for m in matches])
            
            # Compute homography and store homography matrix in h
            h,_ = cv2.findHomography(raw_pts, target_pts, cv2.RANSAC, ransacReprojThreshold=ransac_threshold, confidence=homography_confidence)
            # Calculate the determinant of the top left 2x2 cells of the homography matrix 
            # to check it's stability
            h_det = np.linalg.det(h)
            
            # Store all homography matrices
            h_all[:,:,i] = h
            # Store all h_det values
            h_det_all = np.append(h_det_all,np.array(h_det))
            
            # Best match is defined by the homography matrix determinant being closest
            # to one, this is one is selected from h_all to perform warpPerspective
            best_h_det = np.argmin(abs(1-h_det_all))
            best_h = h_all[:,:,best_h_det]
        
            # Update progress bar after each iteration
            pbar.update(1)
        
    best_match_tar = tar_list[best_h_det]   
    
    im_tar = cv2.imread(os.path.join(targetDir,best_match_tar))
        
    # Perform registration
    reg_im = cv2.warpPerspective(newIm,best_h,(im_tar.shape[1],im_tar.shape[0]))    
    
    # Tell which target image resulted in h_det closest to 1
    print('\nBest match with %s.' % best_match_tar)
    
    return reg_im, best_match_tar