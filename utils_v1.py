# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:37:10 2017

@author: yang
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

#get all image in the given directory persume that this directory only contain image files
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

#function take the chess board image and return the object points and image points
 #相机的畸变系数调整
    #读入图片，预处理图片，检测交点，标定相机
    #FindChessboardCorners是opencv的一个函数，可以用来寻找棋盘图的内角点位置。
    #使用cv2.calibrateCamera()进行标定，这个函数会返回标定结果、相机的内参数矩阵、畸变系数、旋转矩阵和平移向量
def calibrate(images,grid=(9,6)):
    object_points=[]
    img_points = []
    for img in images:
        #np.mgrid 行列等差数列
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)
    return object_points,img_points
#透视矩阵
def get_M_Minv():
    #while
    #src= np.array([[416, 95], [882, 82], [5, 716], [1277, 356]], dtype="float32")
    #dst = np.array([[395.,87.], [883.,83.], [11., 715.], [1005., 624.]], dtype="float32")
    # black
    # src= np.array([[519, 30], [928, 40], [42, 700], [1274, 302]], dtype="float32")
    # dst = np.array([[515.,24.], [927.,43.], [44., 696.], [943., 429.]], dtype="float32")

    #华为视频的透视
    src = np.array([[11, 558], [225, 431], [994, 434], [1270, 560]], dtype="float32")
    dst = np.array([[87., 699.], [23., 267.], [1105., 216.], [1063., 704.]], dtype="float32")
    # src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    # dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    #cap.jpg
    # src = np.float32([[(125 , 535), (421, 343), (840,346), (1270, 536)]])
    # dst = np.float32([[(201, 712), (151, 116), (1074, 55), (1243, 685)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv
    
#function takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
#标定结果
#进行畸变修正
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst
 #方法的缺陷是在路面颜色相对较浅且车道线颜色为黄色时，无法捕捉到车道线（第三，第六，第七张图），但在其他情况车道线捕捉效果还是不错的。
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output
#全局阈值
#是当路面颜色相对较浅且车道线颜色为黄色时，颜色变化梯度较小，想要把捕捉车道线需要把阈值下限调低，
# 然而这样做同时还会捕获大量的噪音像素，效果会更差。
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
#计算角度
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
# H :色相 S：饱和度  L :亮度
#路面颜色相对较浅且车道线颜色为黄色的区域，车道线仍然被清晰的捕捉到了，然而在其他地方表现却不太理想
def hls_select(img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output
#L*表示物体亮度，u*和v*是色度
def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    # l_channel = luv[:, :, 1]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output
#Lab空间取坐标Lab，其中L亮度；a的正数代表红色，负端代表绿色；b的正数代表黄色， 负端代表兰色(
def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    # b_channel = lab[:,:,2]
    b_channel = lab[:, :, 0]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

def find_line(binary_warped,nwindows, margin, minpix):
    # Take a histogram of the bottom half of the image
    #高度360:720遍历 宽度0:1280
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # plt.plot(histogram)
    # plt.show()
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    #中点
    midpoint = np.int(histogram.shape[0]/2)#720
    #判断是否有左右车道线
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    # nwindows = 9
    # Set height of windows
    #定义窗大小
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    # print('left_',leftx_current)
    rightx_current = rightx_base
    # print('right_',  rightx_current)
    # Set the width of the windows +/- margin
    # margin = 100
    # Set minimum number of pixels found to recenter window
    # minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    #检测出左右车道线
    print('1',leftx_current,rightx_current)
    if leftx_current == 0 and rightx_current == 0:
        return
    if leftx_current !=0 and rightx_current!=0 :
    # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            #判断窗口中非零像素的大小是否符合重新绘制滑动窗口的条件，符合则更新左右滑动窗口位置起点
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Fit a second order polynomial to each
        #np.polyfit函数：采用的是最小二次拟合
        # left_fit = []
        # right_fit = []
        # print(lefty,leftx)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit, left_lane_inds, right_lane_inds,out_img
    #只检测出左车道线
    if leftx_current !=0 and rightx_current ==0:
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            # 判断窗口中非零像素的大小是否符合重新绘制滑动窗口的条件，符合则更新左右滑动窗口位置起点
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        out_img[lefty, leftx] = [255, 0, 0]


        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = []
        # right_fit = np.array([0,0,0], dtype='float')
        return left_fit, right_fit, left_lane_inds, right_lane_inds,out_img
    #只检测出右车道线
    if leftx_current ==0 and rightx_current !=0:
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            # 判断窗口中非零像素的大小是否符合重新绘制滑动窗口的条件，符合则更新左右滑动窗口位置起点
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        right_lane_inds = np.concatenate(right_lane_inds)

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        out_img[righty, rightx] = [0, 0, 255]

        # Fit a second order polynomial to each
        # np.polyfit函数：采用的是最小二次拟合
        left_fit = []
        # right_fit = []
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit, left_lane_inds, right_lane_inds,out_img




def find_lane_pixels(binary_warped, nwindows, margin, minpix):
    # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    histogram = np.sum(binary_warped[:binary_warped.shape[0] // 2, :], axis=0)

    #方法一
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    leftx_current = leftx_base
    print('left',leftx_current)
    rightx_current = rightx_base
    print('right', rightx_current )
    if abs((leftx_base-midpoint) - (midpoint-rightx_base))< 100:
           leftx_current = leftx_base
           rightx_current = rightx_base
    if leftx_base<100 and rightx_base <1100:
        leftx_base = np.argmax(histogram[100:midpoint])
        if abs((leftx_base - midpoint) - (midpoint - rightx_base)) < 100:
            leftx_current = leftx_base
            rightx_current = rightx_base
    if leftx_base>100 and rightx_base >1100:
        rightx_base = np.argmax(histogram[midpoint:1100]) + midpoint
        if abs((leftx_base - midpoint) - (midpoint - rightx_base)) < 100:
            leftx_current = leftx_base
            rightx_current = rightx_base
    if leftx_base<100 and rightx_base >1100:
        leftx_base = np.argmax(histogram[100:midpoint])
        rightx_base = np.argmax(histogram[midpoint:1100]) + midpoint
        if abs((leftx_base - midpoint) - (midpoint - rightx_base)) < 100:
            leftx_current = leftx_base
            rightx_current = rightx_base
    if leftx_base>100 and rightx_base <1100:
        # leftx_base = np.argmax(histogram[100:midpoint])
        # rightx_base = np.argmax(histogram[midpoint:1100]) + midpoint
        if abs((leftx_base - midpoint) - (midpoint - rightx_base)) < 100:
            leftx_current = leftx_base
            rightx_current = rightx_base




    # ind_left = np.argpartition(histogram[:midpoint], -2)[-2:]  # 取前两个
    # ind_right = np.argpartition(histogram[midpoint:], -2)[-2:] + midpoint  # 取前两个
    # print(ind_left,ind_right)
    # print(histogram[ind_left],histogram[ind_right] )
    # if histogram[ind_left[0]] > histogram[ind_left[1]]:
    #     left_max = ind_left[0]
    #     left_secondmax = ind_left[1]
    # else:
    #     left_max = ind_left[1]
    #     left_secondmax = ind_left[0]
    # if histogram[ind_right[0]] > histogram[ind_right[1]]:
    #     right_max = ind_right[0]
    #     right_secondmax = ind_right[1]
    # else:
    #     right_max = ind_right[1]
    #     right_secondmax = ind_right[0]
    # #1:1
    # first_long = abs(left_max-right_max)
    # #1:2
    # second_long = abs(left_max-right_secondmax )
    # #2:1
    # third_long = abs(left_secondmax - right_max)
    # # 2:2
    # fourst_long = abs(left_secondmax - right_secondmax)
    # # leftx_base = np.argmax(histogram[:midpoint])
    # # rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # if first_long <=750:
    #     leftx_base = left_max
    #     rightx_base = right_max
    # if second_long <= 750:
    #     leftx_base = left_max
    #     rightx_base = right_secondmax
    # if third_long <=750:
    #     leftx_base = left_secondmax
    #     rightx_base = right_max
    # if fourst_long <=750:
    #     leftx_base = left_secondmax
    #     rightx_base = right_secondmax
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # midpoint = np.int(histogram.shape[0] // 2)
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    # leftx_current = leftx_base
    # rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
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

    return leftx, lefty, rightx, righty, out_img
# 利用上次保存的前十帧的均值作为本次左右车道线
def find_line_by_previous(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()#找出非零的像素点位置
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    # print('left_find', left_fit)
    # print('right_find', right_fit)
    if left_fit != [] and right_fit != []:
        #左车道线 二项式拟合的左右偏差±100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit, left_lane_inds, right_lane_inds
    if left_fit == [] and right_fit != []:
        # 左车道线 二项式拟合的左右偏差±100
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                         right_fit[1] * nonzeroy + right_fit[2] + margin)))
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        right_fit = np.polyfit(righty, rightx, 2)
        left_lane_inds = []
        # left_fit = []

        return left_fit, right_fit, left_lane_inds, right_lane_inds
    if left_fit != [] and right_fit == []:
        # 左车道线 二项式拟合的左右偏差±100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                       left_fit[1] * nonzeroy + left_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_lane_inds = []
        return left_fit, right_fit, left_lane_inds, right_lane_inds
    if left_fit == [] and right_fit == []:
        print('no line')

#使用逆变形矩阵把鸟瞰二进制图检测的车道镶嵌回原图，并高亮车道区域:
def draw_area(undist,binary_warped,Minv,left_fit, right_fit):
    # Generate x and y values for plotting
    #等差数列 0-719 720个
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #二项式
    # print('le',left_fit )
    # print('re', right_fit)
    if left_fit != [] and right_fit !=[]:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        #合并三通道
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))


        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result,color_warp,newwarp
    if left_fit != [] and right_fit == []:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        # 合并三通道
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        # pts = np.hstack((pts_left))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts_left]), (0, 255, 0))
        # color_warp[pts_left] = (0,255,0)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result, color_warp, newwarp
    if left_fit == [] and right_fit != []:
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        # 合并三通道
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        # pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts_right]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result, color_warp, newwarp
    if left_fit == [] and right_fit == []:
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        # 合并三通道
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result, color_warp, newwarp
#计算车道曲率及车辆相对车道中心位置
# 利用检测车道得到的拟合值(find_line 返回的left_fit, right_fit)计算车道曲率，及车辆相对车道中心位置：
def calculate_curv_and_pos(binary_warped,left_fit, right_fit):
    # Define y-value where we want radius of curvature
    #等差数列
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    if left_fit != [] and right_fit !=[]:
        #二项式拟合
        leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        curvature = ((left_curverad + right_curverad) / 2)
        #print(curvature)
        lane_width = np.absolute(leftx[719] - rightx[719])
        lane_xm_per_pix = 3.7 / lane_width
        veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
        cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
        #车辆相对中心位置
        distance_from_center = cen_pos - veh_pos
        return curvature,distance_from_center
    if left_fit == [] and right_fit != []:
        # 二项式拟合
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        curvature = ((right_curverad))
        distance_from_center = 0
        return curvature, distance_from_center
    if left_fit != [] and right_fit == []:
        # 二项式拟合
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        curvature = ((left_curverad ) )
        # print(curvature)
        # 车辆相对中心位置
        distance_from_center = 0
        return curvature, distance_from_center

def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask

def select_white(image):
    lower = np.array([170,170,170])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    
    return mask
#使用"cv2.putText()"方法处理原图展示车道曲率及车辆相对车道中心位置信息：
def draw_values(img,curvature,distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm"%(round(curvature))
    
    if distance_from_center>0:
        pos_flag = 'right'
    else:
        pos_flag= 'left'
        
    cv2.putText(img,radius_text,(100,100), font, 1,(255,255,255),2)
    center_text = "Vehicle is %.3fm %s of center"%(abs(distance_from_center),pos_flag)
    cv2.putText(img,center_text,(100,150), font, 1,(255,255,255),2)
    return img
def thresholding(img):
    #setting all sorts of thresholds
    # x_thresh = abs_sobel_thresh(img, orient='x', thresh_min=35 ,thresh_max=120)
    #mag_thresh1 = mag_thresh(img, sobel_kernel=9, mag_thresh=(45, 150))
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    # hls_thresh = hls_select(img,channel='s',thresh=(200, 255))
    # lab_thresh = lab_select(img, thresh=(155, 200))
    luv_thresh = luv_select(img, thresh=(250, 255))
    x_thresh = 1
    hls_thresh = 0
    mag_thresh1 = 0
    # dir_thresh = 0
    lab_thresh = 0
    # mag_thresh1 = 0
    #Thresholding combination
    # threshholded = np.zeros_like(x_thresh)
    threshholded = np.zeros_like(luv_thresh)
    # threshholded[((x_thresh == 1) & (mag_thresh1 == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    threshholded[((x_thresh == 1) & (mag_thresh1 == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    return threshholded
def fit_polynomial(img,Minv,binary_warped, nwindows=9, margin=100, minpix=50):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels( binary_warped, nwindows, margin, minpix)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    # 合并三通道
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # out_img[left_fitx] = [255, 0, 0]
    # out_img[right_fitx] = [0, 0, 255]
    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return color_warp,out_img, left_fit, right_fit, ploty
def do_segment(frame):#分割
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    print('heigh',height) #480
   # print ( '1', frame.shape[1] 854
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    # 720 1280
    # polygons = np.array([
    #                         [(540, 0), (540,720), (1279, 0)]#找出三个点，根据车固定的宽度
    #                     ])
    polygons = np.array([
        [(0,50), (0, 640), (1280, 600), (1120, 40)]  # 找出三个点，根据车固定的宽度
    ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, 255)#三角形区域全部填充为1
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)#按位与 只要画线三角形区域内的点
    # print('se',segment)
    # cv2.imshow('seg',segment)
    return segment
def computer_img_mean(img):
    x = np.array(img)
    mpv = x.mean(axis=(0, 1))  # 对于图像的三个通道分别计算平均值，
    mpv_three = x.mean()
    # print('mpv', mpv, x.shape)
    #print('mpv', mpv_three)
    return mpv_three
def my_ap_way(landetctect_index,pixelY,pixelX,roadWidth):
    a2, a1, a0 = np.polyfit(pixelY, pixelX, 2)  # 采用二次多项式拟合非零像素点（车道线）， 可画出此多项式曲线，之后看看怎么显示在当前帧上

    frontDistance = np.argsort(pixelY)[int(len(pixelY) / 6)]  # 这步是为了得到aimLanP，设定目标点距离的参数，这个参数对于车辆行走有影响，需要对应现场调到合适的参数
    aimLaneP = [pixelX[frontDistance], pixelY[frontDistance]]
    # 计算aimLaneP处斜率，从而得到目标点的像素坐标
    lane_Pkx = 2 * a2 * aimLaneP[0] + a1
    # lane_Pky = 2 * a2 * aimLaneP[1] + a1
    #print('a2',a2)
    print('640-landetctect_index',640-landetctect_index)
    k_ver = - 1 / lane_Pkx
    #print('k_ver',k_ver)
    # k_ver = - 1 / lane_Pkx
    theta = abs(math.atan(k_ver))  # 利用法线斜率求aP点坐标
    LorR = 0.8
    aP = [0.0, 0.0]
    #右边的
    if int(750-landetctect_index) < 0:
        if lane_Pkx > 0 :
            aP[0] = aimLaneP[0] + 3*math.cos(theta) * (-LorR) * roadWidth / 2
            aP[1] = aimLaneP[1] + 2*math.sin(theta) * (LorR) * roadWidth / 2
            #if int(640-landetctect_index) < -400:
                #aP[0] = aimLaneP[0] + 2*math.cos(theta) * (LorR) * roadWidth / 2
                #aP[1] = aimLaneP[1] + math.sin(theta) * (-LorR) * roadWidth / 2
        else:
            aP[0] = aimLaneP[0] + 3*math.cos(theta) * (-LorR) * roadWidth / 2
            aP[1] = aimLaneP[1] + 2*math.sin(theta) * (-LorR) * roadWidth / 2
            #if int(640-landetctect_index) < -400:
                #aP[0] = aimLaneP[0] + 2*math.cos(theta) * (LorR) * roadWidth / 2
                #aP[1] = aimLaneP[1] + math.sin(theta) * (LorR) * roadWidth / 2
    if int(530-landetctect_index) > 0:
        if lane_Pkx > 0 :
            aP[0] = aimLaneP[0] + 3*math.cos(theta) * (LorR) * roadWidth / 2
            aP[1] = aimLaneP[1] + 2*math.sin(theta) * (-LorR) * roadWidth / 2
            #if int(640-landetctect_index) < 400:
                #aP[0] = aimLaneP[0] + 2*math.cos(theta) * (-LorR) * roadWidth / 2
                #aP[1] = aimLaneP[1] + math.sin(theta) * (LorR) * roadWidth / 2
        else:
            aP[0] = aimLaneP[0] + 3*math.cos(theta) * (LorR) * roadWidth / 2
            aP[1] = aimLaneP[1] + 2*math.sin(theta) * (LorR) * roadWidth / 2
            #if int(640-landetctect_index) < 400:
               # aP[0] = aimLaneP[0] + 2*math.cos(theta) * (-LorR) * roadWidth / 2
               # aP[1] = aimLaneP[1] + math.sin(theta) * (-LorR) * roadWidth / 2
    return aP[0],aP[1]
def  aimpoin(show_img,leftx,lefty,left_fit,landetctect_index):
    #左车线的点
    x_cmPerPixel = 90 / 665.00  # ？？？？？？？？
    y_cmPerPixel = 81 / 680.00
    roadWidth = 750

    y_offset = 50.0  # cm
    n = 0.6
    # 轴间距
    I = 58.0 *n
    # 摄像头坐标系与车中心间距
    D = 18.0 *n
    # 计算cmdSteer的系数
    k = -19 * n

    # aP = [0.0, 0.0]
    # lastP = [0.0, 0.0]
    frontDistance_left = np.argsort(lefty)[int(len(lefty) / 6)] # 这步是为了得到aimLanP，设定目标点距离的参数，这个参数对于车辆行走有影响，需要对应现场调到合适的参数
    aimLaneP_left = [leftx[ frontDistance_left], lefty[ frontDistance_left]]
    #右车线的点
    # frontDistance_right = np.argsort(righty)[int(len(righty) / 8)]
    # aimLaneP_right = [rightx[frontDistance_right], righty[frontDistance_right]]

    #计算斜率
    # lane_Pkx_left = 2*left_fit[0]*aimLaneP_left[0]+left_fit[1]
    # lane_Pky_left = 2 * left_fit[0] * aimLaneP_left[1] + left_fit[1]
    # lane_Pkx_right = 2 * right_fit[0] * aimLaneP_right[0] + right_fit[1]
    # lane_Pky_right = 2 * right_fit[0] * aimLaneP_right[1] + right_fit[1]
    ap0 , ap1 = my_ap_way(landetctect_index,lefty, leftx, roadWidth)
    # if left_fit[0] >= 0:
    #     #45度为分界点
    #     if lane_Pky_left >0 and lane_Pkx_left >0 :
    #         if  lane_Pky_left > lane_Pkx_left:
    #             lanePk_left = lane_Pkx_left
    #         else:
    #             lanePk_left =  lane_Pky_left
    #     else:
    #             lanePk_left = lane_Pkx_left
    #
    # if left_fit[0] < 0:
    #     if lane_Pky_left > 0 and lane_Pkx_left > 0:
    #         if  lane_Pky_left > lane_Pkx_left:
    #             lanePk_left =  lane_Pky_left
    #         else:
    #             lanePk_left = lane_Pkx_left
    #     else:
    #             lanePk_left = lane_Pky_left
    #
    # k_ver = - 1 / lanePk_left
    # # 雷达衔接处的斜率缓冲区
    # # self.lanpk_before.append(lanePk)
    # LorR = 0.8
    #
    #
    # theta = math.atan(k_ver)  # 利用法线斜率求aP点坐标
    # if (left_fit[0] >= 0):
    #     if (k_ver < 0):
    #
    #         aP[0] = aimLaneP_left[0] + math.cos(theta) * (LorR) * roadWidth / 2
    #         aP[1] = aimLaneP_left[1] + math.sin(theta) * (LorR) * roadWidth / 2
    #         # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
    #     else:
    #         aP[0] = aimLaneP_left[0] + math.cos(theta) * (LorR) * roadWidth / 2
    #         aP[1] = aimLaneP_left[1] + math.sin(theta) * (LorR) * roadWidth / 2
    #         # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
    # else:
    #     if (k_ver < 0):
    #         aP[0] = aimLaneP_left[0] + math.cos(theta) * (-LorR) * roadWidth / 2
    #         aP[1] = aimLaneP_left[1] + math.sin(theta) * (-LorR) * roadWidth / 2
    #         # print('theta=', theta)
    #         # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
    #     else:
    #         aP[0] = aimLaneP_left[0] + math.sin(theta) * (-LorR) * roadWidth / 2
    #         aP[1] = aimLaneP_left[1] + math.cos(theta) * (-LorR) * roadWidth / 2
    #         # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
        # 把aP点可视化
        # 把aimLaneP点可视化
    cv2.circle(show_img, (aimLaneP_left[0], aimLaneP_left[1]), 40, (255, 0, 0), -1)
    # plt.scatter(aimLaneP[0], aimLaneP[1], color='b', s=300)
    cv2.circle(show_img, (int(ap0), int(ap1)), 40, (0, 0, 255), -1)

    # 计算目标点的真实坐标，这里x_cmPerPixel\y_cmPerPixel虽然官方说他们做了几次测试说没必要标定了，但是有条件的可以作🌿一些修正
    # 这里的映射关系还没完全弄懂，到时候可实地勘测一下具体的真实距离关系，再看看有没有必要调整一些参数和偏移量
    ap0 = (ap0 - 599) * x_cmPerPixel
    ap1 = (680 - ap1) * y_cmPerPixel + y_offset

    # 根据pure persuit算法计算实际转角，这里会根据自行车模型作一些修正
    if 530 <= int(640-landetctect_index) <=750 :
        steerAngle = 0
    else :
        steerAngle = math.atan(2 * I * ap0 / (ap0 * ap0 + (ap1 + D) * (ap1 + D)))
    #steerAngle = math.atan(2 * I * ap0 / (ap0 * ap0 + (ap1 + D) * (ap1 + D)))

    # self.cam_cmd.angular.z = k * steerAngle
    angular_z = k * steerAngle
    return angular_z


def find_line_max(binary_warped, nwindows, margin, minpix):
    # Take a histogram of the bottom half of the image
    # 高度360:720遍历 宽度0:1280
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # histogram_x = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram_x = np.sum(binary_warped[int(binary_warped.shape[0] * 9 / 10):, :], axis=0)  # 直方图车道线检测，只检测下半部分图
    lane_base = np.argmax(histogram_x)  # 取最大值作为车道线起始点
    midpoint_x = int(histogram_x.shape[0] / 2)  # 车道线中点，没有车道线的话就取中点
    # plt.plot( histogram_x)
    # plt.show()
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # 中点
    midpoint = np.int(histogram_x.shape[0] / 2)  # 720
    # 判断是否有左右车道线

    # Choose the number of sliding windows
    # nwindows = 9
    # Set height of windows
    # 定义窗大小
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = lane_base
    # print('left_',leftx_current)

    # print('right_',  rightx_current)
    # Set the width of the windows +/- margin
    # margin = 100
    # Set minimum number of pixels found to recenter window
    # minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []

    # 检测出左右车道线
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 3)
        # Identify the nonzero pixels in x and y within the window
        # 判断窗口中非零像素的大小是否符合重新绘制滑动窗口的条件，符合则更新左右滑动窗口位置起点
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        elif window >= 1:  # 如果当前帧超过3个框都没有满足上述阈值，则说明当前帧没有车道线，可调参数
            # self.self.noLanePub.publish(0)
            # self.self.noLanePub.publish(0)
            break

        # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)


    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    out_img[lefty, leftx] = [255, 0, 0]
    if (leftx.size != 0):
        flag = 1
        left_fit = np.polyfit(lefty, leftx, 2)
        # aimpoin(out_img,leftx,lefty,left_fit)
        return left_fit, left_lane_inds, out_img,flag,leftx,lefty,lane_base
    else :
        flag = 2
        left_fit = []
        return left_fit, left_lane_inds, out_img,flag,leftx,lefty,lane_base
    #
    #     # Fit a second order polynomial to each
    #     # np.polyfit函数：采用的是最小二次拟合
    #     # left_fit = []
        # right_fit = []
        # print(lefty,leftx)
    # show_img = cv2.merge((binary_warped, binary_warped, binary_warped))
    # histogram_x = np.sum(binary_warped[int(binary_warped.shape[0] * 9 / 10):, :], axis=0)  # 直方图车道线检测，只检测下半部分图
    # lane_base = np.argmax(histogram_x)  # 取最大值作为车道线起始点
    # midpoint_x = int(histogram_x.shape[0] / 2)  # 车道线中点，没有车道线的话就取中点
    #
    # histogram_y = np.sum(binary_warped[0:binary_warped.shape[0], :], axis=1)  # y轴上下直方图检测
    # histogramy = np.sum(histogram_y[0:720])
    # # midpoint_y = 320  # int(histogram.shape[0]/2)   # 这个320是不是也是人为可调的？
    #
    # # ax1.plot(np.arange(0, 720, 1), histogram_y)
    # # plt.show()
    # # plt.pause(0.0000001)
    # # plt.close()
    #
    # # upper_half_histSum = np.sum(histogram_y[0:midpoint_y])
    # # lower_half_histSum = np.sum(histogram_y[midpoint_y:])
    # # try:
    # #     hist_sum_y_ratio = (upper_half_histSum) / (lower_half_histSum)  # 防止除数为0异常，hist_sum_y_ratio参数有什么用？看下面貌似是用来检测左右车道线的
    # # except:
    # #     hist_sum_y_ratio = 1
    # # print(hist_sum_y_ratio)
    #
    # nwindows = 15
    # window_height = int(binary_warped.shape[0] / nwindows)  # 定义10个车道线检测框
    # nonzero = binary_warped.nonzero()  # 获取非零像素对应坐标
    # nonzeroy = np.array(nonzero[0])  # 获取非零的y坐标
    # nonzerox = np.array(nonzero[1])  # 获取非零的x坐标
    # lane_current = lane_base  # 当前车道线起点
    # margin = 200  # 检测框大小，可调
    # minpix = 25  # 检测框是否有效的阈值
    #
    # lane_inds = []
    # # 滑动窗口
    # for window in range(nwindows):
    #     win_y_low = binary_warped.shape[0] - (window + 1) * window_height
    #     win_y_high = binary_warped.shape[0] - window * window_height
    #     win_x_low = lane_current - margin
    #     win_x_high = lane_current + margin
    #     cv2.rectangle(show_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 3)
    #     good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
    #             nonzerox < win_x_high)).nonzero()[0]  # 筛选好的框，在目标框内有y的非零像素则视为有效框
    #     lane_inds.append(good_inds)
    #     if len(good_inds) > minpix:  # 大于此阈值的最小像素数量，则更新当前车道线位置
    #         lane_current = int(np.mean(nonzerox[good_inds]))  # 取非零像素均值作为新的车道线位置
    #     elif window >= 1:  # 如果当前帧超过3个框都没有满足上述阈值，则说明当前帧没有车道线，可调参数
    #         # self.self.noLanePub.publish(0)
    #         # self.self.noLanePub.publish(0)
    #         break
    #
    # lane_inds = np.concatenate(lane_inds)  # 把所有框连在一起
    # leftx = nonzerox[ lane_inds]
    # lefty = nonzeroy[ lane_inds]
    # show_img[lefty, leftx] = [255, 0, 0]
    # left_fit = np.polyfit(lefty, leftx, 2)
    # return left_fit, lane_inds, show_img
    # return left_fit, left_lane_inds,out_img
def find_line_by_previous_max(binary_warped,left_fit):
    nonzero = binary_warped.nonzero()#找出非零的像素点位置
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    histogram_x = np.sum(binary_warped[int(binary_warped.shape[0] * 9 / 10):, :], axis=0)  # 直方图车道线检测，只检测下半部分图
    lane_base = np.argmax(histogram_x)  # 取最大值作为车道线起始点
    # print('left_find', left_fit)
    # print('right_find', right_fit)

    #左车道线 二项式拟合的左右偏差±100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))


    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    print('left',leftx,lefty)
    if (leftx.size != 0):
        flag = 1
        left_fit1 = np.polyfit(lefty, leftx, 2)
        return left_fit1, left_lane_inds, leftx, lefty,flag,lane_base
    else :
        flag = 2
        left_fit1 = []
        return left_fit1, left_lane_inds, leftx, lefty, flag, lane_base

    # Fit a second order polynomial to each
    # left_fit = np.polyfit(lefty, leftx, 2)


def draw_area_max(undist,binary_warped,Minv,left_fit,leftx,lefty,landetctect_index):
    # Generate x and y values for plotting
    #等差数列 0-719 720个
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # out_img[lefty, leftx] = [255, 0, 0]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #二项式
    # print('le',left_fit )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    #合并三通道
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # color_warp[lefty, leftx] = [255, 0, 0]
    if (leftx.size == 0):
        return

        # flag = 1
    left_fit = np.polyfit(lefty, leftx, 2)
    angular_z = aimpoin(color_warp, leftx, lefty, left_fit,landetctect_index)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])

    pts = pts_left

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,0, 255))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))


    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result,color_warp,newwarp,angular_z
def calculate_curv_and_pos_max(binary_warped,left_fit):
    # Define y-value where we want radius of curvature
    #等差数列
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    #二项式拟合
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    # curvature = ((left_curverad + right_curverad) / 2)
    #print(curvature)
    # lane_width = np.absolute(leftx[719] - rightx[719])
    # lane_xm_per_pix = 3.7 / lane_width
    # veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
    # cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
    # #车辆相对中心位置
    # distance_from_center = cen_pos - veh_pos
    return left_curverad


def draw_values_max(img, curvature):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm" % (round(curvature))

    # if distance_from_center > 0:
    #     pos_flag = 'right'
    # else:
    #     pos_flag = 'left'

    cv2.putText(img, radius_text, (100, 100), font, 1, (255, 255, 255), 2)
    # center_text = "Vehicle is %.3fm %s of center" % (abs(distance_from_center), pos_flag)
    # cv2.putText(img, center_text, (100, 150), font, 1, (255, 255, 255), 2)
    return img
def thresholding_black(img):
    #setting all sorts of thresholds
    # x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=50 ,thresh_max=120)
    #mag_thresh1 = mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 150))
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.9, 1.6))
    # hls_thresh = utils.hls_select(img, thresh=(200,250))
    lab_thresh = lab_select(img, thresh=(50, 200))
    luv_thresh = luv_select(img, thresh=(250, 255))
    x_thresh = 1
    hls_thresh = 0
    mag_thresh1 = 0
    # x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=230)
    # mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    # dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    # hls_thresh = utils.hls_select(img, thresh=(180, 255))
    # # lab_thresh = utils.lab_select(img, thresh=(155, 200))
    # luv_thresh = utils.luv_select(img, thresh=(225, 255))
    # # x_thresh = 1
    # lab_thresh = 0

    #Thresholding combination
    threshholded = np.zeros_like( luv_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh1 == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    # threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ( (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    return threshholded
