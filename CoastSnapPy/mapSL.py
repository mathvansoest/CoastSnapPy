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
import scipy.io
import os
from skimage.measure import profile_line
from scipy import stats
from skimage.filters import threshold_otsu
from skimage.measure import points_in_poly
from skimage.measure import find_contours

class mapSL():
    
        def __init__(self, matfile, CSinput, CSimage, CSrect, path=os.getcwd(), RmBthresh = 10):
            
            # Load x and y coordinates for transects from .mat-file
            transectsMat = scipy.io.loadmat(matfile)['SLtransects']
            self.transectsX = transectsMat['x'][0][0].astype(float)
            self.transectsY = transectsMat['y'][0][0].astype(float)
            
            # P is the array which will contain all RGB data extracted from the sample pixels in the following steps.
            # P[:,0] contains the Red pixel values, P[:,1] contains the G pixel values, P[:,2] contains the B pixel values.
            P = np.empty((0,3))
            
            # The for loop below samples pixels along each transect, extracts their RGB pixel values 
            # and appends these to the variable P.
            for i in range(self.transectsX.shape[1]):
                
                    '''                
                    In the original CoastSnap MATLAB code 3 things are passed to the function "improfile" to sample pixels along the transects:
                      1 - the georectified image;
                      2 - the world coordinates of the image's limits;
                      3 - the world coordinates (x and y) corresponding to the ends of each transect.
                    
                    The equivalent Python function is "profile_line", however, this function does not allow the world coordinates of the image's limits to be passed like is done in MATLAB.
                    This means that instead of using world coordinates to define the transect ends, the indices of the corresponding pixels in the image array are required.
                    See (Heaney, 2021, pages 56 and 57) for full visual explaination.
                    This means only 2 things are passed to profile_line:
                    1 - the georectified image;
                    2 - the indices of the image array corresponding to the ends of each transect.
                    Thus, the world coordinates of the transect endpoints must be converted to equivalent indices of the image array.
                    
                    Note: This conversion will depend on the extremities of the georectified image which are defined in the CoastSnap Database for a particular site. The relevant cells
                    of the Database, under the sub-heading Rectification Settings are:
                                  Xlimit left; Xlimit right; Ylimit lower; Ylimit upper.
                    The numbers used below are the values of the above from the Manly Database.
                    '''
                    ##MATH## Given the info above, the values below should be linked to the Database and not be fixed at the same values as the Manly site.

                    # M1 is the the row index of the trasect start point.
                    M1 = self.transectsY[0,i]
                    # M1 is converted from the world coordinate. In this case the row index is 2*ycoordinate (see (Heaney, 2021, page 57))
                    M1 = (M1*2)
                    # M2 is the the row index of the trasect end point.
                    M2 = self.transectsY[1,i]
                    # M2 is converted from the world coordinate in the same way M1 is.
                    M2 = (M2*2)
                    # N1 is the the column index of the trasect start point.
                    N1 = self.transectsX[0,i]
                    # N1 is converted from the world coordinate. In this case the column index is 2*(xcoordinate + 400) (see (Heaney, 2021, page 57))
                    N1 = ((N1 + 400) * 2)
                    # N2 is the the column index of the trasect end point.
                    N2 = self.transectsX[1,i]
                    # N2 is converted from the world coordinate in the same way N1 is.
                    N2 = ((N2 + 400) * 2)

                    # Now pixels are sampeled along the transects using the function "profile_line"
                    prof = profile_line(CSrect.im, (M1, N1), (M2, N2),mode = 'constant')
                    # Append the ssample pixel data to the array P
                    P = np.append(P, prof, axis = 0)
            
            #Now a probabilty distribution is generated for the set of pixel values

            # RmBsample is an array of red minus blue values (RmB) for the sampled pixels. This is essentially a filter to emphasise the difference in pixel colour between wet and dry pixels.
            RmBsample = P[: , 0] - P[: , 2]
            # Using this set of RmB values in RmBsample, generate a probability density estimate.
            kde = stats.gaussian_kde(RmBsample)
            # pdf_locs is the range for which values of the probability density estimate will be generated.
            pdf_locs = np.linspace(RmBsample.min(), RmBsample.max(), 400, endpoint=True)
            # pdf_values contains the values of the probabilty density estimate for the range pdf_locs. Essentially a probabilty density function has been determined.
            pdf_values = kde(pdf_locs)
            
            # Now the threshold RmB value representing the shoreline is determined:

            # threshold_otsu determines the RmB value which is the threshold between 'wet' and 'dry' pixels.
            thresh_otsu = threshold_otsu(RmBsample)
            # thres_weightings are the wightings given to each the peak 'wet' and 'dry' RmB values to determine the shoreline RmB value.
            thresh_weightings = [1/3, 2/3]
            # I1 determines the indices of 'wet' RmB values in pdf_locs.
            I1 = np.argwhere(pdf_locs < thresh_otsu)
            # J1 determines the index of the peak 'wet' RmB value (the most frequent in the probability distribution).
            J1 = np.argmax(pdf_values[I1])
            # I2 determines the indices of 'dry' RmB values in pdf_locs.
            I2 = np.argwhere(pdf_locs > thresh_otsu)
            # J2 determines the index of the peak 'dry' RmB value (the most frequent in the probability distribution).
            J2 = np.argmax(pdf_values[I2])
            
            # The RmB value of the 'wet' peak is determined:
            RmBwet = pdf_locs[I1[J1,0]]
            # The RmB value of the 'dry' peak is determined:
            RmBdry = pdf_locs[I2[J2,0]]
            
            # The RmB value of the shoreline is calculated by assigning the wightings to each peak.
            thresh = thresh_weightings[0]*RmBwet + thresh_weightings[1]*RmBdry
            
            # Convert the image data type to float to enable the following step to operate as intended.
            Iplan = CSrect.im.astype("float")
            # RminusBdouble is an array corresponding to the image however instead of containing RGB for each pixel, there is a single RmB vslue for each:
            RminusBdouble = Iplan[:,:,0] - Iplan[:,:,2]
            
            # Define the region of interest using the coordinates of the transect starts and ends:
            # ROIx contains the x-coordinates of the vertices counding the region of interest.
            ROIx = np.concatenate((self.transectsX[0,:], np.flipud(self.transectsX[1,:])))
            # ROIy contains the y-coordinates of the vertices counding the region of interest.
            ROIy = np.concatenate((self.transectsY[0,:], np.flipud(self.transectsY[1,:])))
            
            # Now mask the region outside of the region of interest:

            # The required variables need to be organised before being passed into the function "points_in_poly"
            # All x-grid coodinates of the georectified image are flattened into a 1d array
            XFlat = CSinput.Xgrid.flatten()
            # All y-grid coodinates of the georectified image are similarly flattened into a 1d array
            YFlat = CSinput.Ygrid.flatten()
            # Combine XFlat and YFlat to generate the 2-columned array, points. It contains the x and y (world) coordinates of all the pixels in the image.
            points = np.column_stack((XFlat, YFlat))
            # The vertice data the region of interest a combined into one array too.
            verts = np.column_stack((ROIx, ROIy))
            
            # Now the region of interest is isolated by masking the region outside:

            # The indices of the region to be masked are determined
            Imask = ~points_in_poly(points, verts)
            Imask = np.reshape(Imask,[CSinput.Xgrid.shape[0],CSinput.Xgrid.shape[1]])
            # The pixels to be masked are assigned the NaN value.
            RminusBdouble[Imask] = np.nan
            
            # Now locate the contours of pixels containing the RmB threshold corresponding to the shoreline.

            # thresh is the RmB threshold
            # c holds the location data of the contours
            c = find_contours(RminusBdouble,thresh)
            
            # The are multiple contours identified to the one corresponding to the shoreline is identified as the longest.
            # c_lenths is created to hold the data about the number of verties in each contour and hence their lengths.
            c_lengths = np.empty(0)
            
            # Find the number of vertices in each of the contours and add this information to the variable c_lengths.
            for i in range(len(c)):
                c_lengths = np.append(c_lengths, len(c[i]))
            
            # Determine the longest contour.
            longest_contour_loc = np.argmax(c_lengths)
            
            # Extract the location data from the longest contour.
            xyz_x = c[longest_contour_loc][:,1]
            xyz_y = c[longest_contour_loc][:,0]
            
            # Convert from pixel coords into grid coordinates
            xyz_x = xyz_x*(np.array(abs(CSinput.xlim[0]-CSinput.xlim[1]))/CSrect.im.shape[1])+np.array(CSinput.xlim[0])
            xyz_y = xyz_y*(np.array(abs(CSinput.ylim[0]-CSinput.ylim[1]))/CSrect.im.shape[0])
            
            # Stack coordinates
            slpoints = np.vstack((xyz_x,xyz_y)).T
            
            # Initialize empty arrays
            slx = np.zeros((1,self.transectsX.shape[1]))
            sly = np.zeros((1,self.transectsY.shape[1]))
            angle =np.empty(slx.shape)
            
            # Select most shoreward RmB threshold if multiple are detected along transect
            for i in range(slx.shape[1]):
                
                angle = np.arctan(np.diff(self.transectsY[:,i]/np.diff(self.transectsX[:,i])))
                anglemat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])[:,:,0]
                slpoints_new = slpoints - np.matlib.repmat([self.transectsX[0,i], self.transectsY[0,i]], slpoints.shape[0], 1)
                points_rot = slpoints_new@anglemat
                max_distance = np.sqrt(np.diff(self.transectsY[:,i])**2+np.diff(self.transectsX[:,i])**2)
            
                I = np.array(np.where((points_rot[:,1]>-1) & (points_rot[:,1]<1) & (points_rot[:,0]>0) & (points_rot[:,0]<max_distance)))
                
                if  np.array(I).size == 0:
                    I = float("NaN")
                else:    
                    Imin = np.argmin(points_rot[I,0])
                    slx[0,i]= slpoints[I[0,Imin],0]
                    sly[0,i]= slpoints[I[0,Imin],1]
                    
            self.x = slx
            self.y = sly
            self.UTMx = slx + CSinput.x0
            self.UTMy = sly + CSinput.y0
            self.xymat = {'xyz': np.hstack([np.rot90(slx), np.rot90(sly)])}