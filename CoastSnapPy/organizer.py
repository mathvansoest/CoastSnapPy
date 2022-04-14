# CoastSnap Python: file and directory organizer class
"""
This class will allow users CoastSnap images to be organized and allows the directory
structure to be check if all necesarry paths exist. 
    
Created by: Math van Soest
"""

import os
import numpy as np
from datetime import datetime
import pandas as pd
import time
from shutil import copy
import cv2
import scipy.io

from pathlib import Path

class organizer():
    
    def __init__(self, OrImageName,site,outputPath=None,CSpath = os.path.dirname(os.getcwd())):
        self.site = site
        self.pathCS = CSpath
        self.outputPath = outputPath
        self.OrImageName = OrImageName

        # Read xlsx CoastSnapPy Database
        xl = pd.ExcelFile(os.path.join(self.pathCS,'Database','CoastSnapPyDB.xlsx'))
        # Get sheetnames and define the sites to this
        self.sites = xl.sheet_names
        # Work from main CoastSnap folder
        os.chdir(self.pathCS)
        
    def check_directories(self):
        
        if self.outputPath == None:
            self.outputLocation = ""
            self.outputDirName = 'Images'
        else:
            self.outputLocation = self.outputPath
            self.outputDirName = 'CSoutput'
            
        # First layer of directories
        os.makedirs(os.path.join(self.outputLocation,self.outputDirName),exist_ok=True)
        os.makedirs('Shorelines',exist_ok=True)
        os.makedirs('Database',exist_ok=True)
        os.makedirs('CoastSnapPy',exist_ok=True)
        os.makedirs('Objects',exist_ok=True)
        os.makedirs('Target',exist_ok=True)
        
        # Define folders that need site specific subfolders
        siteFolders = 'Shorelines', 'Objects', 'Target', self.outputDirName
        
        # Define subfolders for image parent folder
        imageFolders = 'Processed','Raw','Rectified','Registered','Detected','Plots'
        
        # Create Site Specific subfolders
        for siteFolder in siteFolders:
            for site in self.sites:
                
                sitePath = os.path.join(self.pathCS, siteFolder, site)
                sitePathIm =  os.path.join(self.outputLocation, siteFolder, site)
                
                if siteFolder == 'Shorelines':                   
                    os.makedirs(os.path.join(sitePath, self.year), exist_ok=True)
                    os.makedirs(os.path.join('Shorelines','Transects'), exist_ok=True)
                if siteFolder == 'Objects' or 'TargetImages':                    
                    os.makedirs(sitePath, exist_ok=True)
                if siteFolder == self.outputDirName:
                    for imageFolder in imageFolders:                        
                        os.makedirs(os.path.join(sitePathIm, imageFolder, self.year), exist_ok=True)
                        
    
    def gen_paths(self):
        
        # Generate base path
        base = self.pathCS
        
        if self.outputPath == None:
            baseIm = base
        else:
            baseIm = self.outputLocation
        
        # Define all paths necessary to execute CoastSnapPy
        self.pathDB = os.path.join(base, 'Database', 'CoastSnapPyDB.xlsx')
        self.pathIm = os.path.join(baseIm, self.outputDirName, self.site, 'Processed', self.year)
        self.pathImRaw = os.path.join(baseIm, self.outputDirName, self.site, 'Raw', self.year)
        self.pathReg =  os.path.join(baseIm, self.outputDirName, self.site, 'Registered' + self.year)
        self.pathRect = os.path.join(baseIm, self.outputDirName, self.site, 'Rectified', self.year)
        self.pathDetect = os.path.join(baseIm, self.outputDirName, self.site, 'Detected', self.year)
        self.pathPlot = os.path.join(baseIm, self.outputDirName, self.site, 'Plots', self.year)
        self.pathSL = os.path.join(base, 'Shorelines', self.site, self.year)
        self.pathObjects = os.path.join(base, 'Objects', self.site)
        self.pathTarget = os.path.join(base, 'Target', self.site)
        self.pathTrans = os.path.join(base, 'Shorelines','Transects')
        self.fileTrans = os.path.join(self.pathTrans, 'SLtransects_' + self.site + '.mat')
        
        
    def check_time(self):
        #TODO use more elaborte times from image file name
        
        # Get current time data
        now = datetime.now()
        current_time = now.strftime('.%A.%B.%d_%H.%M.%S_%Y.')
        self.year = now.strftime('%Y')
        
        # Get current epoch time
        EpochTime = np.round(time.time()).astype(int)
        
        # Create new file name
        self.file_id = np.str(EpochTime) + current_time + self.site
        
    def check_filename(self):
        
        # Check if file contains keyword .snap
        keyword = '.snap'
        if keyword in str(self.OrImageName):
            
            now = datetime.now()
            self.year = now.strftime('%Y')
            
            # Split filename at keyword
            file_split = str(Path(self.OrImageName).name).split('.snap')
            # Derive file_id and extension
            self.file_id = file_split[0]
            self.file_extension = file_split[1]
            
        else:
            self.check_time()
                          
    def process_new_image(self,rename=False):
        
        # Copy original file to raw folder
        copy(self.OrImageName, self.pathImRaw)

        # Copy original file to processed folder using name convention
        self.NewImageName = self.file_id + '.snap' + '.jpg'

        destination = Path(self.pathIm) / Path(self.NewImageName)
        copy(self.OrImageName, destination)
        
        if rename == True:
            os.rename(self.OrImageName,self.NewImageName)
        
    def process_output_files(self, output_file, output_type):
        
        # Define options for output_type
        output_type_options = ['.reg','.rect','.SL','.Plot']
        # Check if user defined output_type correctly
        check_count = output_type_options.count(output_type)
        if check_count < 1:
            print('output_type should be either ' + str(output_type_options))
            return
        
        paths = self.pathReg, self.pathRect, self.pathSL, self.pathPlot
        
        if output_type != '.SL':
            NewFileName = self.file_id + output_type + '.jpg'
            cv2.imwrite(os.path.join(paths[output_type_options.index(output_type)], NewFileName), output_file)
        elif output_type == '.SL': 
            NewFileName = self.file_id + output_type + '.mat'
            scipy.io.savemat(os.path.join(paths[output_type_options.index(output_type)], NewFileName), output_file)
