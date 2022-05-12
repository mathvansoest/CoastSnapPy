# CoastSnap Python: function for image rectification
"""
This function will allow users CoastSnap images that have already been registered
to rectify the image into a plan view image. The information from the CSinput class
and the CSimage class are to be used for this script to work.

Created by: Math van Soest
"""

import numpy as np
import numpy.matlib
from scipy import interpolate
from scipy.optimize import curve_fit
from angles2R import angles2R

class rectification():
    
    def __init__(self,CSinput,CSimage,UV,registeredIm=False):
        self.UV = UV
        self.CSinput = CSinput
        self.CSimage = CSimage
        self.z = 0
        
        
        # Check if UV is in the correct input format
        if self.UV.shape != (2,len(CSinput.GCPsCombo)):
            raise ValueError('The UV input should be an np.array with shape 2 x nGCPs, with x values on row-1 and y-values on row-2.')
        
        def findUV3DOF(xyz, beta3, beta4, beta5):
            
            #note: scipy.optimize.curve_fit required the parameters to fit as separate arguments in
            #the function definition. This is why beta1, beta2 etc... are written rather than just beta.
        
            K = np.array([[fx, 0, c0U],[0, -fy, c0V],[0, 0, 1]]).astype(float)
    
            R = angles2R(beta3, beta4, beta5)
        
            I = np.eye(3)
            C = self.CSinput.beta0[0:3]
            #to use np.hstack in the next cell, the arrays I and C both need to be 2D:
            C.shape = (3,1)
        
            IC = np.hstack((I,-C))
        
            P = np.matmul(np.matmul(K,R),IC)
            #note when comparing the output to P in MATLAB, the 1st and 3rd entries in the bottom
            #row are too small to show up in the MATLAB way of displaying the data. The results are
            #the same
        
            P = P/P[2,3]
        
            #note: instead of np.tranpose, an alternative is to place ".T" after the object to be transposed.
            #I keep np.transpose since this makes is more obvious to someone analysing the code.
            UV = np.matmul(P,np.vstack((np.transpose(xyz), np.ones((1, len(xyz)), dtype = float))))
        
            UV = UV/np.matlib.repmat(UV[2,:],3,1)
            UV = np.transpose(np.concatenate((UV[0,:], UV[1,:])))
            return UV 
        
        def findUV6DOF(xyz, beta0, beta1, beta2, beta3, beta4, beta5):
        
            K = np.array([[fx, 0, c0U],[0, -fy, c0V],[0, 0, 1]]).astype(float)
            
            R = angles2R(beta3, beta4, beta5)
        
            I = np.eye(3)
            C = np.array([beta0, beta1, beta2]).astype(float)
            #to use np.hstack in the next cell, the arrays I and C both need to be 2D:
            C.shape = (3,1)
        
            IC = np.hstack((I,-C))
        
            P = np.matmul(np.matmul(K,R),IC)
            #note when comparing the output to P in MATLAB, the 1st and 3rd entries in the bottom
            #row are too small to show up in the MATLAB way of displaying the data. The results are
            #the same
        
            P = P/P[2,3]
          
            #note: instead of np.tranpose, an alternative is to place ".T" after the object to be transposed.
            #I keep np.transpose since this makes is more obvious to someone analysing the code.
            UV = np.matmul(P,np.vstack((np.transpose(xyz), np.ones((1, len(xyz)), dtype = float))))
            
            UV = UV/np.matlib.repmat(UV[2,:],3,1)
            UV = np.transpose(np.concatenate((UV[0,:], UV[1,:])))
            return UV
        
        def onScreen(U, V, Umax, Vmax):
                
            Umin = 1
            Vmin = 1

            #Column of zeros, same length as UV(from the grid) (ie one for each coord set)
            yesNo = np.zeros((len(U),1))
            #Gives at 1 for all the UV coords which have existing corresponding pixel values from the oblique image
            on = np.where((U>=Umin) & (U<=Umax) & (V>=Vmin) & (V<=Vmax)) [0]
            yesNo[on] = 1
            return yesNo
        
        # Check if original or registered image are used
        if registeredIm == False:
            NU = self.CSimage.NU
            NV = self.CSimage.NV
            c0U = NU/2
            c0V = NV/2
            im = self.CSimage.color

        else:
            NV, NU = self.CSimage.reg.shape[:2]
            c0U = NU/2
            c0V = NV/2
            im = self.CSimage.reg
        
        # Set up array for defining the focal length that will be assessed
        A = np.arange(5, 500005, 5)
        B = np.arange(5, 500005, 5)
        
        fx_max = 0.5*NU/np.tan(self.CSinput.FOV[0]*np.pi/360) #From Eq. 4 in Harley et al. (2019)
        fx_min = 0.5*NU/np.tan(self.CSinput.FOV[1]*np.pi/360) #From Eq. 4 in Harley et al. (2019)
        fx_min = interpolate.interp1d(A, B, kind='nearest')(fx_min)
        fx_max = interpolate.interp1d(A, B, kind='nearest')(fx_max)
        
        fx_all = np.arange(fx_min, fx_max+5, 5)
        fy_all = np.copy(fx_all)
        
            
        xyz= np.array(self.CSinput.GCPmat[['x','y','z']])
    
        mse_all = np.zeros(len(fx_all))
        
        nGCP = len(self.CSinput.GCPsCombo)
    
        UV_true = np.concatenate(self.UV)
    
        
        for i in range(len(fx_all)):
        
            fx = fx_all[i].astype(float)
            fy = fy_all[i].astype(float)
            beta3, Cov = curve_fit(findUV3DOF, xyz, UV_true, self.CSinput.beta0[3:6], maxfev=4000)
            UV_pred = findUV3DOF(xyz, beta3[0], beta3[1], beta3[2])
            mse_all[i] = np.mean((UV_true-UV_pred)**2)*((2*nGCP)/((2*nGCP)-len(beta3)))
            #mse_all[i] = mean_squared_error(UV_true, UV_pred)*((2*nGCPs)/((2*nGCPs)-len(CameraBeta)))
        
        fx = fx_all[np.argmin(mse_all)].astype(float)
        fy = fy_all[np.argmin(mse_all)].astype(float)
        beta3, Cov = curve_fit(findUV3DOF, xyz, UV_true, self.CSinput.beta0[3:6])
        
        self.beta6 = np.hstack([self.CSinput.beta0[0:3],beta3])
        UV_pred = findUV6DOF(xyz, self.beta6[0], self.beta6[1], self.beta6[2], self.beta6[3], self.beta6[4], self.beta6[5])  
        
        # Reshape UV_pred and UV_true
        self.UV_pred = np.reshape(UV_pred,[2,nGCP])
        UV_true = np.reshape(UV_true,[2,nGCP])
  
        leny, lenx = self.CSinput.Xgrid.shape
        #Zeros array with same dimensions as the grid and depth of 3 for each of RGB values
        images_sumI = np.zeros([leny,lenx,3])
        #Zeros array with 2D grid dimensions
        images_N = np.zeros(self.CSinput.Xgrid.shape)
        
        #Create an array representing the (x,y,z) coordinates of every point in the grid. shape: (nPoints, 3)
        xyz = np.column_stack((self.CSinput.Xgrid.T.flatten(), self.CSinput.Ygrid.T.flatten(), np.matlib.repmat(self.z, len(self.CSinput.Xgrid.T.flatten()), 1)))
        
        #All xyz coordinates in the grid are given their corresponding UV coordinates which are rounded to nearest integer
        UV = findUV6DOF(xyz, self.beta6[0], self.beta6[1], self.beta6[2], self.beta6[3], self.beta6[4], self.beta6[5])

        UV = np.around(UV.astype('float'))
        
        #Order='F' means the reshape is done column by column.
        UV = np.reshape(UV, (-1, 2), order='F')
        
        #Gives the index of all the UV coords which will have pixel data
        good = np.where(onScreen(UV[:,0], UV[:,1], NU, NV) == 1)[0]
        
        UV = UV.astype(int)
        #arr is an array containing the useful UV coordinates with the V coords on top of the U coords
        arr = np.array([UV[good,1], UV[good,0]])
        #returns the linear indices of the usful coordinates
        ind = np.ravel_multi_index(arr, (NV, NU), mode='clip', order='F')
        
        #same as before size of grid with 3 depth for RGB
        foo = images_sumI
        
        for i in range(3):
            #Take each of the RGB information arrays
            I3 = im[:,:,i]
            #Reshape it so that it is one column (column by column)
            I3 = np.reshape(I3, (-1, 1), order='F')
            #Extract the data from the image pixels corresponding to xyz grid points
            I2 = I3[ind]
            #Make bar the same shape as the real-world grid
            bar = foo[:,:,i]
            #Straignten it out
            bar = np.reshape(bar, (-1, 1), order='F')
            #The relevant coordinates of bar (real-world grid) will now attain the pixel data
            bar[good] = I2
            #Reshape bar
            bar = np.reshape(bar, (len(images_sumI), -1), order='F')
            #Put the data back into foo to hold it safely
            foo[:,:,i] = bar
        
        #Pixel info in the grid
        images_sumI  = foo

        images_N = np.reshape(images_N, (-1, 1), order='F')
        #Puts a 1 in every location where there is pixel data in the grid
        images_N[good] = 1
        images_N = np.reshape(images_N, (self.CSinput.Xgrid.shape[0], self.CSinput.Xgrid.shape[1], 1), order='F')
        
        #Copies N to fit the shape of the RGB grid
        N = np.tile(images_N, (1, 1, 3))
        
        #Replace all zeros with NaN
        N[N==0]=np.nan
        
        #All grid coordinates without pixel data are assigned a NaN value due to the division by 0 from N:
        self.im = (images_sumI/N).astype(np.uint8)