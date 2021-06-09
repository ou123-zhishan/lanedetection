import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np
import line
import imageio
import pipeline
imageio.plugins.ffmpeg.download()
left_line = line.Line()
right_line = line.Line()
cal_imgs = utils.get_images_by_dir('camera_cal')
# print('',cal_imgs)
M,Minv = utils.get_M_Minv()
object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
#较黑暗的图片 用亮度去检测车道线 还有用全局阈值 例如H55视频 lab mag luv 才有用 b_channel = lab[:, :, 2]

test_imgs = utils.get_images_by_dir('test_image5')
undistorted = []
j=0
plt.figure(figsize=(100,100))
mpv = np.zeros(shape=(3,))
for img in test_imgs:
    j = 0
    # mpv = np.zeros(shape=(3,))
    img = utils.cal_undistort(img,object_points,img_points)
    print('img',img.shape,img)
    filepath = 'F:/path/lanedetection_21_04_07/test_image5/cap1.jpg'
    mpv_channel = utils.computer_img_mean(img)
    if mpv_channel >=70.0:
        all_thresh = utils.thresholding(img)
    else:
        all_thresh = pipeline.thresholding(img)
    origin_thr = np.zeros_like(all_thresh)
    origin_thr[(all_thresh >= 1)] = 255
    cv2.imshow('origin_thr',origin_thr)
    cv2.imwrite(filepath, origin_thr )


    # x = np.array(img)
    # mpv = x.mean(axis=(0,1))#对于图像的三个通道分别计算平均值，
    # print('mpv',mpv,x.shape)
    # cv2.calcHist(img,0,8,256)
    #h55
    # all_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=120)
    # all_thresh = utils.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 150))
    # all_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    # all_thresh = utils.hls_select(img, channel='s',thresh=(200,255))
    # all_thresh = utils.lab_select(img, thresh=(50, 200))
    # all_thresh = utils.luv_select(img, thresh=(80, 255))
    # all_thresh = utils.thresholding(img)
    #h1.avi
    # all_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=30, thresh_max=120)#可以不用
    # all_thresh = utils.mag_thresh(img, sobel_kernel=9, mag_thresh=(35, 150))
    # all_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    # all_thresh = utils.hls_select(img, channel='s',thresh=(200,255)) #不用
    # all_thresh = utils.lab_select(img, thresh=(155, 200)) #不用
    # all_thresh = utils.luv_select(img, thresh=(250, 255))
    # all_thresh = utils.thresholding(img)
    # all_thresh = pipeline.thresholding(img)
    # thresholding_wrape0 = utils.do_segment(all_thresh)
    # all_thresh = pipeline.thresholding(img)
    thresholding_wrape0 = cv2.warpPerspective(origin_thr, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    cv2.imshow('thresholding_wrape0',thresholding_wrape0)
    cv2.waitKey()
    #检测出左右车道线
    if (left_line.detected and right_line.detected) or(left_line.detected == False and right_line.detected)\
            or (left_line.detected and right_line.detected == False):
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholding_wrape0,
                                                                left_line.current_fit,right_line.current_fit)
    else:
        #拟合车道线
        left_fit, right_fit, left_lane_inds, right_lane_inds,out_img= utils.find_line(thresholding_wrape0,nwindows=9,
                                                                               margin=100, minpix=50)
        # color_warp, out_img, left_fit, right_fit, ploty = utils.fit_polynomial(img, Minv, thresholding_wrape0,
        #                                                                        nwindows=9, margin=100, minpix=50)
    left_line.update(left_fit)
    right_line.update(right_fit)
    # # left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholding_wrape,nwindows=9, margin=100, minpix=50 )
    result,out_img1,out_img0 = utils.draw_area(img,thresholding_wrape0,Minv,left_fit, right_fit)
    # color_warp,out_img, left_fit, right_fit, ploty   = utils.fit_polynomial(img,Minv,thresholding_wrape0, nwindows=9, margin=100, minpix=50)

    # histogram = np.sum( thresholding_wrape0[thresholding_wrape0.shape[0] // 2:, :], axis=0)
    histogram = np.sum(thresholding_wrape0[thresholding_wrape0.shape[0] // 2:, :], axis=0)
    # ind_left = np.argpartition(histogram[:midpoint],-2)[-2:]#取前两个
    # ind_right = np.argpartition(histogram[midpoint:], -2)[-2:] + midpoint # 取前两个
    # if histogram[ind_left[0]] > histogram[ind_left[1]]:
    #     left_max = ind_left[0]
    #     left_secondmax = ind_left[1]
    # else:
    #     left_max = ind_left[1]
    #     left_secondmax = ind_left[2]
    # if histogram[ind_right[0]] > histogram[ind_right[1]]:
    #     right_max = ind_right[0]
    #     right_secondmax = ind_right[1]
    # else:
    #     right_max = ind_right[1]
    #     right_secondmax = ind_right[2]

    print(histogram.shape,type(histogram ))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    print(all_thresh.shape)
    print('histogram',histogram)
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])#返回最大值索引 从0：开始
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholding_wrape)


    plt.subplot(7, 1, j+1)
    plt.imshow(img[:,:,[2,1,0]])
    plt.subplot(7, 1, j+2)
    plt.imshow(all_thresh , cmap='gray')
    plt.plot(histogram)
    plt.subplot(7, 1, j+3)
    # plt.imshow( thresholding_wrape,cmap='gray')
    plt.plot(histogram)
    plt.subplot(7, 1, j + 4)
    plt.imshow(thresholding_wrape0,cmap='gray')
    # plt.plot(histogram)
    plt.subplot(7, 1, j + 5)
    # plt.imshow(color_warp ) plt.subplot(8, 1, j + 7)
    # K-近邻
    plt.imshow(out_img1)
    # plt.subplot(8, 1, j + 6)
    # plt.imshow(newwarp)

    plt.subplot(7, 1, j + 6)
    plt.imshow(out_img0)
    plt.subplot(7, 1, j + 7)
    plt.imshow(out_img)
    plt.show()
    undistorted.append(img)