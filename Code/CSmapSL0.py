# CoastSnap Python: function for shoreline mapping
"""
This function will allow users CoastSnap images to map the current shoreline position
from images that have already been rectified. It returns the x and y coordinates of the 
pixel values where the shoreline is detected along each of the specified transects 
as defined in the .mat file. 

Created by: Math van Soest
"""
import numpy as np
import numpy.matlib
from CSreadDB import CSinput
from CSreadIm import CSim
from CSrectification import CSrectification
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io
import os
from skimage.measure import profile_line
from scipy import stats
from skimage.filters import threshold_otsu
from skimage.measure import points_in_poly
from skimage.measure import find_contours

class CSmapSL():
    
        def __init__(self, matfile, CSinput, CSimage, CSrect, path=os.getcwd(), RmBthresh = 10):
            
            # Load x and y coordinates for transects from .mat-file
            transectsMat = scipy.io.loadmat(matfile)['SLtransects']
            self.transectsX = transectsMat['x'][0][0].astype(float)
            self.transectsY = transectsMat['y'][0][0].astype(float)
            
            #P contains all sample data
            P = np.empty((0,3))
            
            # Sample pixels at transects to determine threshold
            for i in range(self.transectsX.shape[1]):
                
                    M1 = self.transectsY[0,i]
                    M1 = (M1*2)
                    M2 = self.transectsY[1,i]
                    M2 = (M2*2)
                    N1 = self.transectsX[0,i]
                    N1 = ((N1 + 400) * 2)
                    N2 = self.transectsX[1,i]
                    N2 = ((N2 + 400) * 2)
                    prof = profile_line(CSrect.im, (M1, N1), (M2, N2),mode = 'constant')
                    P = np.append(P, prof, axis = 0)
            
            RmBsample = P[: , 0] - P[: , 2]
            kde = stats.gaussian_kde(RmBsample)
            pdf_locs = np.linspace(RmBsample.min(), RmBsample.max(), 400, endpoint=True)
            
            pdf_values = kde(pdf_locs)
            
            thresh_otsu = threshold_otsu(RmBsample)
            thresh_weightings = [1/3, 2/3]
            I1 = np.argwhere(pdf_locs < thresh_otsu)
            J1 = np.argmax(pdf_values[I1])
            I2 = np.argwhere(pdf_locs > thresh_otsu)
            J2 = np.argmax(pdf_values[I2])
            
            RmBwet = pdf_locs[I1[J1,0]]
            RmBdry = pdf_locs[I2[J2,0]]
            
            thresh = thresh_weightings[0]*RmBwet + thresh_weightings[1]*RmBdry
            
            Iplan = CSrect.im.astype("float")
            RminusBdouble = Iplan[:,:,0] - Iplan[:,:,2]
            
            ROIx = np.concatenate((self.transectsX[0,:], np.flipud(self.transectsX[1,:])))
            ROIy = np.concatenate((self.transectsY[0,:], np.flipud(self.transectsY[1,:])))
            
            XFlat = CSinput.Xgrid.flatten()
            YFlat = CSinput.Ygrid.flatten()
            points = np.column_stack((XFlat, YFlat))
            verts = np.column_stack((ROIx, ROIy))
            
            Imask = ~points_in_poly(points, verts)
            Imask = np.reshape(Imask,[CSinput.Xgrid.shape[0],CSinput.Xgrid.shape[1]])
            
            RminusBdouble[Imask] = np.nan
            
            #defining thresh the same as that in MATLAB for testing
            # thresh = RmBthresh
            
            c = find_contours(RminusBdouble,thresh)
            
            c_lengths = np.empty(0)
            
            for i in range(len(c)):
                c_lengths = np.append(c_lengths, len(c[i]))
            
            longest_contour_loc = np.argmax(c_lengths)
            
            xyz_x = c[longest_contour_loc][:,1]
            xyz_y = c[longest_contour_loc][:,0]
            
            # Convert from pixel coords into grid coordinates
            xyz_x = xyz_x*(np.array(abs(CSinput.xlim[0]-CSinput.xlim[1]))/CSrect.im.shape[1])+np.array(CSinput.xlim[0])
            xyz_y = xyz_y*(np.array(abs(CSinput.ylim[0]-CSinput.ylim[1]))/CSrect.im.shape[0])
            
            slx = np.zeros((1,self.transectsX.shape[1]))
            sly = np.zeros((1,self.transectsY.shape[1]))
            slpoints = np.vstack((xyz_x,xyz_y)).T
            
            angle =np.empty(slx.shape)
            
            for i in range(slx.shape[1]):
                
                angle = np.arctan(np.diff(self.transectsY[:,i]/np.diff(self.transectsX[:,i])))
                anglemat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])[:,:,0]
                slpoints_new = slpoints - np.matlib.repmat([self.transectsX[0,i], self.transectsY[0,i]], slpoints.shape[0], 1)
                points_rot = slpoints_new@anglemat
                max_distance = np.sqrt(np.diff(self.transectsY[:,i])**2+np.diff(self.transectsX[:,i])**2)
            
                I = np.array(np.where((points_rot[:,1]>-1) & (points_rot[:,1]<1) & (points_rot[:,0]>0) & (points_rot[:,0]<max_distance)))
                
                if  np.array(I).size == 0:
                    print('I = empty')
                else:    
                    Imin = np.argmin(points_rot[I,0])
                    slx[0,i]= slpoints[I[0,Imin],0]
                    sly[0,i]= slpoints[I[0,Imin],1]
                    
            self.x = slx
            self.y = sly
            self.xymat = {'x':slx,
                          'y':sly}
            
    
if __name__ == '__main__':
    
    # Define path to location of the CoastSnap Database
    path = r'C:\Coastal Citizen Science\CoastSnap\Database\CoastSnapDB.xlsx'
    sitename = 'egmond'
 
    CSinput = CSinput(path, sitename)
    
    UV = np.array([[2683.07992933693,1994.84531801943,2462.97897751479,],
                   [961.109189295402,1148.93483322320,990.960816201514]])
    
    imname = '1583622000.Sun.Mar.08_00_00_00.CET.2020.egmond.snap.FroukjeHajer.jpg'
    
    im = CSim(imname)
    
    rect = CSrectification(CSinput, im, UV)
    
    SL = CSmapSL('SLtransects_egmond.mat',CSinput,im,rect)
    
    plt.figure(11)
    plt.plot(SL.x[0,:],SL.y[0,:], color = 'r')
    plt.imshow(rect.im, extent = (min(CSinput.x), max(CSinput.x), max(CSinput.y), min(CSinput.y)))
    plt.xlabel("M")
    plt.ylabel("N")
    plt.gca().invert_yaxis()
    plt.title("Corrected Shoreline (Python)")