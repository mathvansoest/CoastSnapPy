# CoastSnap Python: Read CoastSnap Database as defined for the original Matlab code
"""
This script creates classes for all relevant data from the CoastSnap database; such as camera position, azimuth, tilt, roll etc..

Image database: C:\Coastal Citizen Science\CoastSnap\Database\CoastSnapDB.xlsx

Created by: Math van Soest
"""

import pandas as pd
import numpy as np

class CSDB:
    
    def __init__(self, path, sitename):
        self.sitename = sitename
        self._parse_file(path)
    
    def _parse_file(self, path):
        xl_db = pd.ExcelFile(path)
        self.data = xl_db.parse(self.sitename)

class CSinput(CSDB):
    
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
        
        
if __name__ == '__main__':         
    path = r'C:\Coastal Citizen Science\CoastSnap\Database\CoastSnapDB.xlsx'
    sitename = 'egmond'
    xl_db = pd.ExcelFile('C:\Coastal Citizen Science\CoastSnap\Database\CoastSnapDB.xlsx')
    # xl_site = xl_db.parse(sitename)
    
    EgmondData = CSinput(path, sitename)
    
    # print(EgmondData.GCPmat)
    print(EgmondData.GCPmat)
