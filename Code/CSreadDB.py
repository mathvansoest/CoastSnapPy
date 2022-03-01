# CoastSnap Python: Read CoastSnap Database as defined for the original Matlab code
"""
This script creates classes for all relevant data from the CoastSnap database; such as camera position, azimuth, tilt, roll etc..

Image database: .\CoastSnap\Database\CoastSnapDB.xlsx

NOTE: It is important to note that this class has been defined for a fixed xcel-file format
that means that the exact lay-out should be used for adding new sites.

Created by: Math van Soest
"""

import pandas as pd
import numpy as np
import glob
import os
import cv2
import openpyxl
import pyperclip as pc
import matplotlib.pyplot as plt
from CSgetUV import getUV

class CSreadDB:
    
    def __init__(self, path, sitename):
        self.path = path
        self.sitename = sitename
        self._parse_file(path)
    
    def _parse_file(self, path):
        xl_db = pd.ExcelFile(self.path)
        self.all_sites = xl_db.sheet_names
        self.data = xl_db.parse(self.sitename)
        self.data2 = self.data.set_index('Station Data')
    
    @property
    def x0(self):
        x0 = self.data.iloc[0,1]
        return x0
    
    @property
    def y0(self):
        y0 = self.data.iloc[1,1]
        return y0
    
    @property
    def z0(self):
        z0 = self.data.iloc[2,1]
        return z0
    
    @property
    def azimuth(self):
        azimuth = self.data.iloc[14,1]
        return azimuth
    
    @property
    def tilt(self):
        tilt = self.data.iloc[15,1]
        return tilt
    
    @property
    def roll(self):
        roll = self.data.iloc[16,1]
        return roll
    
    @property
    def xlim(self):
        xlim = np.array([self.data.iloc[9,1],self.data.iloc[10,1]])
        return xlim

    @property
    def ylim(self):
        ylim = np.array([self.data.iloc[11,1],self.data.iloc[12,1]])
        return ylim
    
    @property
    def dxdy(self):
        dxdy = self.data.iloc[13,1]
        return dxdy
    
    @property
    def beta0(self):
        beta0 = np.array([0,0,self.z0,self.azimuth,self.tilt,self.roll])
        return beta0
    
    @property
    def x(self):
        x = np.arange(self.xlim[0],self.xlim[1]+self.dxdy,self.dxdy)
        return x
    
    @property
    def y(self):
        y = np.arange(self.ylim[0],self.ylim[1]+self.dxdy,self.dxdy)
        return y

    @property
    def Xgrid(self):
            Xgrid, Ygrid = np.meshgrid(self.x,self.y)
            return Xgrid
    
    @property
    def Ygrid(self):
            Xgrid, Ygrid = np.meshgrid(self.x,self.y)
            return Ygrid
    
    @property
    def iGCPs(self):
         # Locate 'GCP Name' in excel file
        iGCPs = self.data[self.data.iloc[:,0] == 'GCP name'].index
        return iGCPs
    
    @property
    def nGCPs(self):
        # Define amount of GCPs specified in site excel sheet
        nGCPs = len(self.iGCPs)
        return nGCPs
    
    @property
    def GCPsCombo(self):
        # Get user defined combination of GCPs to be used
        iGCPsCombo = [self.data[self.data.iloc[:,0] == 'GCP combo'].index.values]
        GCPsCombo = self.data.iloc[iGCPsCombo[0][0],1]
        GCPsCombo = np.array(GCPsCombo[1:-1].split()).astype(int)-1
        return GCPsCombo
    
    @property
    def GCPsName(self):
        GCPsName = self.data.iloc[self.iGCPs,1]
        GCPsName = GCPsName.values.tolist()
        return GCPsName
    
    @property
    def GCPmat(self):
        # Initialize empty dataframe with column names
        GCPmat = pd.DataFrame(columns=['easting', 'northing', 'x', 'y','z'])
        
        # Fill GCPmat with location data of GCPs
        for i in range(len(self.GCPsCombo)):
            iGCP = self.iGCPs[self.GCPsCombo[i]]
            
            GCPmat.loc[i,'easting'] = self.data.iloc[iGCP+1,1]
            GCPmat.loc[i,'northing'] = self.data.iloc[iGCP+2,1]
            GCPmat.loc[i,'z'] = self.data.iloc[iGCP+3,1]
            GCPmat.loc[i,'x'] = GCPmat.loc[i,'easting']-self.x0
            GCPmat.loc[i,'y'] = GCPmat.loc[i,'northing']-self.y0
            
        return GCPmat
    
    @property
    def FOV(self):
        FOV = np.zeros(2)
        FOV[0] = self.data.iloc[17,1]
        FOV[1] = self.data.iloc[18,1]
        return FOV
    
    @property
    def ObjectNames(self):
        ObjectNames = self.data2.loc['Object Names'].iloc[0]
        return ObjectNames.split(',')
    
    @property
    def ObjectModels(self):
        ObjectModels = self.data2.loc['Models'].iloc[0]
        return ObjectModels.split(',')
    
    @property
    def RefImage(self):
        RefImage = self.data2.loc['Reference Image'].iloc[0]
        return RefImage
    
    def getUV(self,targetIm):
        
        target_index = self.data[self.data.iloc[:,0]==targetIm].index
        
        UVx = self.data2.iloc[target_index+1]
        UVy = self.data2.iloc[target_index+2]
        
        UVx = UVx.iloc[:,:len(self.GCPsCombo)]
        UVy = UVy.iloc[:,:len(self.GCPsCombo)]
        
        UV = np.vstack([UVx,UVy])
        
        if UV.shape != (2,len(self.GCPsCombo)):
            raise ValueError('Check if target image in DB has the same filename as images in target folder')
        
        return UV
    
    def checkUV(self,targetDir):
        """"
        This function checks if all target images have specified UV points.
        If not, each will have show up and a small gui will direct you to 
        provide them. 
        """
        # Get target images
        jpg_extension = targetDir + "\*.jpg"
        
        # List all target and mask files
        target_list = [os.path.basename(x) for x in glob.glob(jpg_extension)]
        
        # See which target images exist in database
        db_list = self.data2.index[self.data.iloc[:,0].isin(target_list)].tolist()
        
        # Compare if each of the images files has a corresponding mask file
        missing_targets = list(set(target_list).difference(db_list))
        
        UVxl = pd.DataFrame()
        
        if len(missing_targets) >= 1:
        
            for tar in reversed(missing_targets):
                # Get UVpoints from missing target image
                UV = getUV(targetDir,tar,np.array(self.GCPsName)[self.GCPsCombo])
                
                # Add UVx and UVy
                UVxl_new = np.hstack([[['UVx'],['UVy']],UV])
                # Add target_image file name 
                UVxl_new = np.vstack([np.hstack([np.array([tar]),np.tile(np.nan,3)]),UVxl_new])
                
                UVxl = UVxl.append(pd.DataFrame(UVxl_new))            
            
            UVxl.to_clipboard(index=False,header=False)
            
            
            print('New UV points have been copied on clipboard and should be pasted in your xcel-database')
            
            return UVxl
            
            exit()
            
        else:
            print('All target images have UV points specified in DataBase')
        
if __name__ == '__main__':
    
    db = CSreadDB('C:\Github\CoastSnap\Database\CoastSnapDB.xlsx','egmond')
    
    UV = db.checkUV('C:\Github\CoastSnap\Target\egmond')

    