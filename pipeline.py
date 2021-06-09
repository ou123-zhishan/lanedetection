# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: yang
"""

import os
import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np
import line
import imageio
imageio.plugins.ffmpeg.download()
# from moviepy.editor import VideoFileClip
import moviepy.editor as mpy

# imageio.plugins.ffmpeg.download()

def thresholding(img):
    #setting all sorts of thresholds
    # x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=50 ,thresh_max=120)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 150))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.9, 1.6))
    # hls_thresh = utils.hls_select(img, thresh=(200,250))
    lab_thresh = utils.lab_select(img, thresh=(50, 200))
    luv_thresh = utils.luv_select(img, thresh=(250, 255))
    x_thresh = 1
    hls_thresh = 0
    # x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=230)
    # mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    # dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    # hls_thresh = utils.hls_select(img, thresh=(180, 255))
    # # lab_thresh = utils.lab_select(img, thresh=(155, 200))
    # luv_thresh = utils.luv_select(img, thresh=(225, 255))
    # # x_thresh = 1
    # lab_thresh = 0

    #Thresholding combination
    threshholded = np.zeros_like( mag_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    # threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ( (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    return threshholded


def processing(img,object_points,img_points,M,Minv,left_line,right_line):
    #camera calibration, image distortion correction
    undist = utils.cal_undistort(img,object_points,img_points)
    #get the thresholded binary image
    mpv_channel = utils.computer_img_mean( undist)
    if mpv_channel >= 70.0:
        thresholded = utils.thresholding( undist)
    else:
        thresholded = thresholding( undist)
    # thresholded = thresholding(undist)
    #perform perspective  transform
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    #perform detection
    # if left_line.detected and right_line.detected:
    #     left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    # else:
    #     #拟合车道线
    #     left_fit, right_fit, left_lane_inds, right_lane_inds,out_img0 = utils.find_line(thresholded_wraped,nwindows=9, margin=100, minpix=50)
    # left_line.update(left_fit)
    # right_line.update(right_fit)
    if (left_line.detected and right_line.detected) or (left_line.detected == False and right_line.detected) \
            or (left_line.detected and right_line.detected == False):
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,
                                                                                           left_line.current_fit,
                                                                                           right_line.current_fit)
    else:
        # 拟合车道线
        left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = utils.find_line(thresholded_wraped, nwindows=9,
                                                                                        margin=100, minpix=50)
        # color_warp, out_img, left_fit, right_fit, ploty = utils.fit_polynomial(img, Minv, thresholding_wrape0,
        #                                                                        nwindows=9, margin=100, minpix=50)
    left_line.update(left_fit)
    right_line.update(right_fit)

    #draw the detected laneline and the information
    area_img,color_warp,newwarp= utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    curvature,pos_from_center = utils.calculate_curv_and_pos(thresholded_wraped,left_fit, right_fit)
    result = utils.draw_values(area_img,curvature,pos_from_center)

    return result

#
#
# left_line = line.Line()
# right_line = line.Line()
# cal_imgs = utils.get_images_by_dir('camera_cal')
# # # print('',cal_imgs)
# # M,Minv = utils.get_M_Minv()
# object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
# test_imgs = utils.get_images_by_dir('test_images')
# undistorted = []
# j=0
# plt.figure(figsize=(100,100))
# for img in test_imgs:
#     j = 0
#     img = utils.cal_undistort(img,object_points,img_points)
    # all_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=80)
    # all_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    # all_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.9, 1.6))
    # all_thresh = utils.hls_select(img, channel='l',thresh=(180, 253))
    # all_thresh = utils.lab_select(img, thresh=(150, 2))
    # all_thresh = utils.luv_select(img, thresh=(220, 252))
    # all_thresh = thresholding(img)
    # thresholding_wrape = cv2.warpPerspective(all_thresh, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    # left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholding_wrape,nwindows=9, margin=100, minpix=50 )
    # result,out_img = utils.draw_area(img,thresholding_wrape,Minv,left_fit, right_fit)
    # # out_img, left_fit, right_fit, ploty   = utils.fit_polynomial(thresholding_wrape, nwindows=9, margin=100, minpix=50)
    #
    # histogram = np.sum(thresholding_wrape[thresholding_wrape.shape[0] // 2:, :], axis=0)
    # # Find the peak of the left and right halves of the histogram
    # # These will be the starting point for the left and right lines
    # print('histogram',histogram)
    # midpoint = np.int(histogram.shape[0] / 2)
    # leftx_base = np.argmax(histogram[:midpoint])#返回最大值索引 从0：开始
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholding_wrape)


    # plt.subplot(4, 1, j+1)
    # plt.imshow(img[:,:,[2,1,0]])
    # plt.subplot(4, 1, j+2)
    # plt.imshow(all_thresh , cmap='gray')
    # plt.subplot(4, 1, j+3)
    # plt.imshow( thresholding_wrape,cmap='gray')
    # plt.plot(histogram)
    # plt.subplot(4, 1, j + 4)
    # plt.imshow(out_img)
    # # plt.plot(histogram)
    # plt.show()
    # undistorted.append(img)
 # all_thresh = thresholding(img)
# plt.figure()
# for i in range(len(undistorted)):
#      j = j +1
#      x_thresh = utils.abs_sobel_thresh(undistorted[i], orient='x', thresh_min=35, thresh_max=100)
#      plt.subplot(8,2,j)
#      # plt.imshow(undistorted[i])
#      plt.imshow(x_thresh,cmap='gray')
# plt.show()

# undist = utils.cal_undistort(img,object_points,img_points)
# M,Minv = utils.get_M_Minv()
# left_line = line.Line()
# right_line = line.Line()
# cal_imgs = utils.get_images_by_dir('camera_cal')
# # print('',cal_imgs)
# M,Minv = utils.get_M_Minv()
# object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
# #
# #
# project_outpath = 'vedio_out/project_video_out9.mp4'
# video_input = 'project_video.mp4'
# # project_video_clip = cv2.VideoCapture("project_video.mp4")
#
# project_video_clip = mpy.VideoFileClip('Video_1.avi')
# # project_video_clip = mpy.VideoFileClip('h55.avi ')
# project_video_out_clip = project_video_clip.image_transform(lambda clip: processing(clip,object_points,img_points,M,Minv,left_line,right_line))
# # project_video_clip.reader.close()
# # project_video_clip.audio.reader.close_proc()
# project_video_out_clip.write_videofile(project_outpath, audio=False)


# #draw the processed test image
# test_imgs = utils.get_images_by_dir('test_images')
# undistorted = []
# for img in test_imgs:
#    img = utils.cal_undistort(img,object_points,img_points)
#    undistorted.append(img)
#
# result=[]
# for img in undistorted:
#    res = processing(img,object_points,img_points,M,Minv,left_line,right_line)
#    result.append(res)
#
# plt.figure(figsize=(20,68))
# for i in range(len(result)):
#
#    plt.subplot(len(result),1,i+1)
#    plt.title('thresholded_wraped image')
#    plt.imshow(result[i][:,:,::-1])
# plt.show()
    