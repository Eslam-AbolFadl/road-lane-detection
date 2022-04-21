import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle

%matplotlib inline

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from matplotlib.patches import Polygon
from IPython.display import HTML
from moviepy.editor import VideoFileClip
import glob
import pickle
#%matplotlib inline

import sys
def main():
input=sys.argv[0] # prints python_script.py
output=sys.argv[1] # prints var1
image=sys.argv[3]

#from google.colab import drive
#drive.mount('/content/drive')

#img1=mpimg.imread('/content/drive/MyDrive/Project_data/test_images/straight_lines1.jpg')
#plt.imshow(img1)

def calibrate():
    """
    This function is used to calibrate camera from images of chessboard 
    captured from a camera for which camera matrix and distortion 
    coefficients are to be calculated 
    """
    # Create object points (0 0 0) ... (8 5 0)
    objp = np.zeros((9*6, 3), np.float32) 
    
    # assign grid points to x and y keeping z as zero
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []

    # Get directory for all calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    for indx, fname in enumerate(images):
        
        img = cv2.imread(fname)
        
        # Converting to grayclae - findChessboardCorners expects graysclae image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find inner corners of chess baord
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save camera matrix and distortion coefficient to pickle file
    
    calibration_param = {}
    calibration_param['mtx'] = mtx
    calibration_param['dist'] = dist
    pickle.dump( calibration_param, open('/camera_cal/calibration_pickle.pickle', 'wb') )

calibrate()

def undistort(img, cal_dir = 'camera_cal/calibration_pickle.pickle'):
    """
    'img' is the input image with distortion
    'cal_dir' = path to saved camera matrix and distortion coefficients
    given thses two inputs thia function undistorts the image
    """
    
    # Read saved file
    with open(cal_dir, mode = 'rb') as f:
        file = pickle.load(f)
        
    mtx = file['mtx'] # camera matrix
    dist = file['dist'] # Distortion coefficients
    
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undistorted_img

img = cv2.imread('camera_cal/calibration1.jpg')
undistorted_img = undistort_img(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

ax1.imshow(img)
ax1.set_title('Original image', fontsize = 15)

ax2.imshow(undistorted_img)
ax2.set_title('Undistorted image', fontsize = 15)

def img_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 255)): 
    """
    Given input image 'img' this function first undistorts it and then applies thresholding on 
    the basis if x gradient values(computed using sobel operator) and saturation channel from HSL color space.
    The uppar and lower threshold are given in the argument.
    """
    # Undistort input image
    undistorted_img = undistort(img)
    
    # Convert to HLS color space and separate the all channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h_channel = hls[:, :, 0] # Hue channel
    l_channel = hls[:, :, 1] # Lightness channel
    s_channel = hls[:, :, 2] # Satuaration channel
    
    # Using sobel operator in x direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    
    # Sobel gives -ve derivative when transitoin is from light pixels to dark pixels
    # so taking absolute values which just indicates strength of edge
    abs_sobelx = np.absolute(sobelx)
    
    # For consistency rescaling pixel values by mapping maximum pixel value in image to 255
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack both thresholded images
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    
    # making all the pixel 1 which are either part of x gradient thresholded image or color channel thresholded image
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # To print all different images uncomment below code
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    ax1.imshow(sxbinary, cmap = 'gray')
    ax1.set_title('Sobel x gradient thresholded', fontsize = 15)
    
    ax2.imshow(s_binary, cmap = 'gray')
    ax2.set_title('Saturation channel thresholded', fontsize = 15)
    
    ax3.imshow(color_binary)
    ax3.set_title('Stacked, Green - x gradient, Blue - color threshold', fontsize = 15)
    
    ax4.imshow(combined_binary, cmap = 'gray')
    ax4.set_title('Stacked thresholded', fontsize = 15)
    
    return combined_binary, undistorted_img

_, _ = img_threshold(cv2.imread('/test_images/test2.jpg'))


#Perspective and Inverse perspective transform



def perspective_transform(img):
    
    src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
    dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
    
    img_size = (img.shape[1],img.shape[0])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    
    return warped

def inv_perspective_transform(img):
    
    dst = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
    src = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
    
    img_size = (img.shape[1],img.shape[0])

    # Given src and dst points, calculate the inverse-perspective transform matrix 
    #(which is just perspective transform with src and dst interchanged)
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist


img = cv2.imread('/test_images/test2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
threshold_img, undistorted_img = img_threshold(img)
warped_img = perspective_transform(threshold_img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

ax1.imshow(img)
ax1.set_title('Original image', fontsize = 15)

ax2.imshow(warped_img, cmap='gray')
ax2.set_title('Warped thresholded image', fontsize = 15)

plt.plot(get_hist(warped_img))

def find_lane_pixels(binary_warped):
    """
    only for the first frame of the video this function is run.
    
    input - 'binary_warped' perspective transformed thresholded image
    output - x and y coordinates of the lane pixels
    
    step 1: Get the histogram of warped input image
    step 2: find peaks in the histogram that serves as midpoint for our first window
    step 3: choose hyperparameter for windows
    step 4: Get x, y coordinates of all the non zero pixels in the image
    step 5: for each window in number of windows get indices of all non zero pixels falling in that window
    step 6: Get the x, y coordinates based off of these indices
    """
    # Take a histogram of the bottom half of the image
    histogram = get_hist(binary_warped)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 200

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        # the four boundaries of the window ###
        win_xleft_low   = leftx_current - margin
        win_xleft_high  = leftx_current + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # if the no of pixels in the current window > minpix then update window center
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def fit_poly(leftx, lefty, rightx, righty):
    """
    Given x and y coordinates of lane pixels fir 2nd order polynomial through them
    
    here the function is of y and not x that is
    
    x = f(y) = Ay**2 + By + C
    
    returns coefficients A, B, C for each lane (left lane and right lane)
    """
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def search_around_poly(binary_warped, left_fit, right_fit):
    """
    This function is extension to function find_lane_pixels().
    
    From second frame onwards of the video this function will be run.
    
    the idea is that we dont have to re-run window search for each and every frame.
    
    once we know where the lanes are, we can make educated guess about the position of lanes in the consecutive frame,
    
    because lane lines are continuous and dont change much from frame to frame(unless a very abruspt sharp turn).
    
    This function takes in the fitted polynomial from previous frame defines a margin and looks for non zero pixels 
    in that range only. it greatly increases the speed of the detection.
    """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100
    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # we have left fitted polynomial (left_fit) and right fitted polynomial (right_fit) from previous frame,
    # using these polynomial and y coordinates of non zero pixels from warped image, 
    # we calculate corrsponding x coordinate and check if lies within margin, if it does then
    # then we count that pixel as being one from the lane lines.
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin))).nonzero()[0]
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx = fit_poly(leftx, lefty, rightx, righty)

    return leftx, lefty, rightx, righty

def measure_curvature_real(left_fit_cr, right_fit_cr, img_shape):
    '''
    Calculates the curvature of polynomial functions in meters.
    and returns the position of vehical relative to the center of the lane
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = img_shape[0]
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # It was meentioned in one the course note that camera was mounted in the middle of the car,
    # so the postiion of the car is middle of the image, the we calculate the middle of lane using
    # two fitted polynomials
    car_pos = img_shape[1]/2
    left_lane_bottom_x = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    right_lane_bottom_x = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = ((right_lane_bottom_x - left_lane_bottom_x) / 2) + left_lane_bottom_x
    car_center_offset = np.abs(car_pos - lane_center_position) * xm_per_pix
    
    return (left_curverad, right_curverad, car_center_offset)


def draw_lane(warped_img, undistorted_img, left_fit, right_fit):
    """
    Given warped image and original undistorted original image this function 
    draws final lane on the undistorted image
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, undistorted_img.shape[0]-1, undistorted_img.shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = inv_perspective_transform(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
    
    return result

first_run = True
gleft_fit = gright_fit = None

def Pipeline(img):
    global first_run, gleft_fit, gright_fit
    
    
    threshold_img, undistorted_img = img_threshold(img)
    
    warped_img = perspective_transform(threshold_img)
    
    if first_run:
        leftx, lefty, rightx, righty = find_lane_pixels(warped_img)
        left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)
        gleft_fit = left_fit
        gright_fit = right_fit
        first_run = False
    else:
        leftx, lefty, rightx, righty = search_around_poly(warped_img, gleft_fit, gright_fit)
        left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)
        gleft_fit = left_fit
        gright_fit = right_fit
    
    #print(left_fitx, right_fitx)
    measures = measure_curvature_real(left_fit, right_fit, img_shape = img.shape)
    
    final_img = draw_lane(warped_img, undistorted_img, left_fit, right_fit)
    
    # writing lane curvature and vehical offset on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)
    fontSize = 1
    cv2.putText(final_img, 'Lane Curvature: {:.0f} m'.format(np.mean([measures[0],measures[1]])), 
                (500, 620), font, fontSize, fontColor, 2)
    cv2.putText(final_img, 'Vehicle offset: {:.4f} m'.format(measures[2]), (500, 650), font, fontSize, fontColor, 2)
    
    return final_img

images = glob.glob('/test_images/*.jpg')

for indx, image in enumerate(images):
    plt.imshow(Pipeline(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)))
    plt.show()

from moviepy.editor import VideoFileClip

first_run = True
gleft_fit = gright_fit = None

myclip = VideoFileClip(input)

output_vid = output
clip = myclip.fl_image(Pipeline)
clip.write_videofile(output_vid, audio=False)
myclip.reader.close()
myclip.close()