# CoastSnap Python: file and directory organizer class
"""
This class will allow users CoastSnap images to be organized and allows the directory
structure to be check if all necesarry paths exist. 
    
Created by: Math van Soest
"""

import os
import numpy as np
from datetime import date, datetime
import time
from shutil import copy
import cv2
import scipy.io

class CSorganizer():
    
    def __init__(self, OrImageName,site,CSpath = os.path.dirname(os.getcwd())):
        self.site = site
        self.pathCS = CSpath
        self.OrImageName = OrImageName
        
        # TODO find better way to read sites
        self.sites = 'egmond', 'texel'
        
        # Work from main CoastSnap folder
        os.chdir(self.pathCS)
        
    def check_time(self):
        #TODO use more elaborte times from image file name
        
        # Get current time data
        now = datetime.now()
        current_time = now.strftime('.%A.%B.%d_%H.%M.%S_%Y.')
        self.year = now.strftime('%Y')
        
        # Get current epoch time
        EpochTime = np.round(time.time()).astype(int)
        
        # Create new file name
        self.NewNameTime = np.str(EpochTime) + current_time        
        
    def check_directories(self):
        # First layer of directories
        os.makedirs('Images',exist_ok=True)
        os.makedirs('Shorelines',exist_ok=True)
        os.makedirs('Database',exist_ok=True)
        os.makedirs('Code',exist_ok=True)
        os.makedirs('Objects',exist_ok=True)
        os.makedirs('ReferenceImages',exist_ok=True)
        
        # Define folders that need site specific subfolders
        siteFolders = 'Shorelines', 'Objects', 'ReferenceImages', 'Images'
        
        # Define subfolders for image parent folder
        imageFolders = 'Processed','Raw','Rectified','Registered','Detected'
        
        # Create Site Specific subfolders
        for siteFolder in siteFolders:
            for site in self.sites:
                
                sitePath = self.pathCS + '/' + siteFolder + '/' + site + '/'
                
                if siteFolder == 'Shorelines':                   
                    os.makedirs(sitePath + self.year, exist_ok=True)
                    os.makedirs('Shorelines/Transects', exist_ok=True)
                if siteFolder == 'Objects' or 'ReferenceImages':                    
                    os.makedirs(sitePath, exist_ok=True)
                if siteFolder == 'Images':
                    for imageFolder in imageFolders:                        
                        os.makedirs(sitePath + imageFolder + '/' + self.year, exist_ok=True)
                        
    def gen_paths(self):
        
        # Generate base path
        base = self.pathCS
        
        # Define all paths necessary to execute CoastSnapPy
        self.pathDB = base + '/Database' + '/CoastSnapDB.xlsx'
        self.pathIm = base + '/Images' + '/' + self.site + '/' + 'Processed/' + self.year
        self.pathImRaw = base + '/' + 'Images/' + self.site + '/' + 'Raw/' + self.year
        self.pathReg =  base + '/' + 'Images/' + self.site + '/' + 'Registered/' + self.year
        self.pathRect = base + '/' + 'Images/' + self.site + '/' + 'Rectified/' + self.year
        self.pathDetect = base + '/' + 'Images/' + self.site + '/' + 'Detected/' + self.year
        self.pathSL = base + '/' + 'Shorelines/' + self.site + '/'+ self.year
        self.pathObjects = base + '/' + 'Objects/' + self.site
        self.pathRefIm = base + '/' + 'ReferenceImages/' + self.site
        self.pathTrans = base + '/' + 'Shorelines/Transects/'
        self.fileTrans = self.pathTrans + 'SLtransects_' + self.site + '.mat'
                          
    def process_new_image(self):
        
        #TODO check if processed image already exists
        
        # Copy original file to raw folder
        copy(self.OrImageName, self.pathImRaw)
        
        # Copy original file to processed folder using name convention
        self.NewImageName = self.NewNameTime + 'snap' + '.jpg'
        copy(self.OrImageName, self.pathIm + '/' + self.NewImageName)
        
    def process_output_files(self, output_file, output_type):
        
        # Define options for output_type
        output_type_options = ['reg','rect','SL']
        # Check if user defined output_type correctly
        check_count = output_type_options.count(output_type)
        if check_count < 1:
            print('output_type should be either ' + str(output_type_options))
            return
        
        paths = self.pathReg, self.pathRect, self.pathSL
        
        
        if output_type != 'SL':
            NewFileName = self.NewNameTime + output_type + '.jpg'
            cv2.imwrite(paths[output_type_options.index(output_type)] + '/' + NewFileName, output_file)
        else: 
            NewFileName = self.NewNameTime + output_type + '.mat'
            scipy.io.savemat(paths[output_type_options.index(output_type)] + '/' + NewFileName, output_file)

if __name__ == '__main__':    
    site = 'egmond'
    test = CSorganizer('test1.jpg',site)
    test.check_time()
    test.check_directories()
    test.gen_paths()
    test.process_new_image()
    
    testim = cv2.imread('test1.jpg')
    
    test.process_output_files(testim,'SL')
                
                
                