#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# import rospy
import cv2
import os
import sys
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import utils
import pipeline
import aimpoin_line

plt.style.use('ggplot')  # 使用‘ggplot风格美化图表’

# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge

# 距离映射
x_cmPerPixel = 90 / 665.00#  ？？？？？？？？
y_cmPerPixel = 81 / 680.00
roadWidth = 665
# roadWidth = 6

y_offset = 50.0  # cm

# 轴间距
I = 58.0
# 摄像头坐标系与车中心间距
D = 18.0
# 计算cmdSteer的系数
k = -19


class camera:
    def __init__(self):

        self.camMat = []
        self.camDistortion = []
        self.line_ap = aimpoin_line.Aimpoin_line()
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(r'Video_2.avi')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print('fps',fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)#摄像头分辨率1280*720
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # self.imagePub = rospy.Publisher('images', Image, queue_size=1)
        # self.cmdPub = rospy.Publisher('lane_vel', Twist, queue_size=1)
        # self.noLanePub = rospy.Publisher('no_lane', Int32, queue_size=1)  # 无车道线发布话题
        # self.cam_cmd = Twist()
        # self.cvb = CvBridge()

        # 效果还行
        # src_points = np.array([[11, 558], [225, 431], [994, 434], [1270, 560]], dtype="float32")
        # dst_points = np.array([[114., 699.], [160., 223.], [1272., 299.], [1145., 704.]], dtype="float32")

        # 微调待试验,两边相对对称
        # src_points = np.array([[11, 558], [225, 431], [994, 434], [1270, 560]], dtype="float32")
        # dst_points = np.array([[214., 699.], [160., 223.], [1272., 224.], [1186., 696.]], dtype="float32")

        # 15号版本
        # src_points = np.array([[11, 558], [225, 431], [994, 434], [1270, 560]], dtype="float32")
        # dst_points = np.array([[214., 699.], [60., 243.], [1272., 224.], [1186., 696.]], dtype="float32")
        #透视变换
        src_points = np.array([[11, 558], [225, 431], [994, 434], [1270, 560]], dtype="float32")#源图像中待测矩形的四点坐标
        dst_points = np.array([[214., 699.], [60., 243.], [1272., 224.], [1186., 696.]], dtype="float32")
        # while
        # src_points = np.array([[416, 95], [882, 82], [5, 716], [1277, 356]], dtype="float32")
        # dst_points = np.array([[395., 87.], [883., 83.], [11., 715.], [1005., 624.]], dtype="float32")
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)#返回由源图像中矩形到目标图像矩形变换的矩阵

        self.aP = [0.0, 0.0]
        self.lastP = [0.0, 0.0]
        self.Timer = 0
        self.lanpk_before = []
        self.count = 0


        self.LorR = 0.8  # aimPoint和aP距离偏移量

    def aimpoin(show_img, leftx, lefty, left_fit):
        # 左车线的点
        aP = [0.0, 0.0]
        lastP = [0.0, 0.0]
        frontDistance_left = np.argsort(lefty)[int(len(lefty) / 8)]  # 这步是为了得到aimLanP，设定目标点距离的参数，这个参数对于车辆行走有影响，需要对应现场调到合适的参数
        aimLaneP_left = [leftx[frontDistance_left], lefty[frontDistance_left]]
        # 右车线的点
        # frontDistance_right = np.argsort(righty)[int(len(righty) / 8)]
        # aimLaneP_right = [rightx[frontDistance_right], righty[frontDistance_right]]

        # 计算斜率
        lane_Pkx_left = 2 * left_fit[0] * aimLaneP_left[0] + left_fit[1]
        lane_Pky_left = 2 * left_fit[0] * aimLaneP_left[1] + left_fit[1]
        # lane_Pkx_right = 2 * right_fit[0] * aimLaneP_right[0] + right_fit[1]
        # lane_Pky_right = 2 * right_fit[0] * aimLaneP_right[1] + right_fit[1]

        if left_fit[0] >= 0:

            if lane_Pky_left > lane_Pkx_left:
                lanePk_left = lane_Pkx_left
            else:
                lanePk_left = lane_Pky_left

        if left_fit[0] < 0:
            if lane_Pky_left > lane_Pkx_left:
                lanePk_left = lane_Pky_left
            else:
                lanePk_left = lane_Pkx_left
        k_ver = - 1 / lanePk_left
        # 雷达衔接处的斜率缓冲区
        # self.lanpk_before.append(lanePk)
        LorR = 0.8
        roadWidth = 665
        theta = math.atan(k_ver)  # 利用法线斜率求aP点坐标
        if (left_fit[0] >= 0):
            if (k_ver < 0):

                aP[0] = aimLaneP_left[0] + math.cos(theta) * (LorR) * roadWidth / 2
                aP[1] = aimLaneP_left[1] + math.sin(theta) * (LorR) * roadWidth / 2
                # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
            else:
                aP[0] = aimLaneP_left[0] + math.cos(theta) * (LorR) * roadWidth / 2
                aP[1] = aimLaneP_left[1] + math.sin(theta) * (LorR) * roadWidth / 2
                # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
        else:
            if (k_ver < 0):
                aP[0] = aimLaneP_left[0] + math.cos(theta) * (-LorR) * roadWidth / 2
                aP[1] = aimLaneP_left[1] + math.sin(theta) * (-LorR) * roadWidth / 2
                print('theta=', theta)
                # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
            else:
                aP[0] = aimLaneP_left[0] + math.sin(theta) * (-LorR) * roadWidth / 2
                aP[1] = aimLaneP_left[1] + math.cos(theta) * (-LorR) * roadWidth / 2
                # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
            # 把aP点可视化
            # 把aimLaneP点可视化
        cv2.circle(show_img, (aimLaneP_left[0], aimLaneP_left[1]), 40, (255, 0, 0), -1)
        # plt.scatter(aimLaneP[0], aimLaneP[1], color='b', s=300)
        cv2.circle(show_img, (int(aP[0]), int(aP[1])), 40, (0, 0, 255), -1)

    def __del__(self):
        self.cap.release()
    def ap_way(self,landetctect_index,pixelY,pixelX,roadWidth):
        a2, a1, a0 = np.polyfit(pixelY, pixelX, 2)  # 采用二次多项式拟合非零像素点（车道线）， 可画出此多项式曲线，之后看看怎么显示在当前帧上



        aveX = np.average(pixelX)  # 求曲线x坐标的平均值，对应车道线的中点
        # 找出对应索引
        frontDistance = np.argsort(pixelY)[
            int(len(pixelY) / 8)]  # 这步是为了得到aimLanP，设定目标点距离的参数，这个参数对于车辆行走有影响，需要对应现场调到合适的参数
        aimLaneP = [pixelX[frontDistance], pixelY[frontDistance]]

        # 计算aimLaneP处斜率，从而得到目标点的像素坐标
        # pltx = a2 * aimLaneP[0] * aimLaneP[0] + a1*aimLaneP[0] + a0
        # plty = a2 * aimLaneP[1] * aimLaneP[1] + a1*aimLaneP[1] + a0

        # cv2.imshow('pltx',pltx)
        # cv2.imshow('plty', plty)


        lane_Pkx = 2 * a2 * aimLaneP[0] + a1
        lane_Pky = 2 * a2 * aimLaneP[1] + a1
        print('斜率x', lane_Pkx)
        print('斜率y', lane_Pky)

        if a2 >= 0:

            if lane_Pky > lane_Pkx:
                lanePk = lane_Pkx
            else:
                lanePk = lane_Pky

        if a2 < 0:
            if lane_Pky > lane_Pkx:
                lanePk = lane_Pky
            else:
                lanePk = lane_Pkx

        k_ver = - 1 / lanePk
        print('k_ver', k_ver)

        # 雷达衔接处的斜率缓冲区
        self.lanpk_before.append(lanePk)

        theta = math.atan(k_ver)  # 利用法线斜率求aP点坐标
        print('a2',a2)
        if (a2 >= 0):
            if (k_ver < 0):

                self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
                # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
            else:
                self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
                # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
        else:
            if (k_ver < 0):
                self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.sin(theta) * (-self.LorR) * roadWidth / 2
                print('theta=', theta)
                # self.aP[1] = k_ver * (self.aP[0] - aimLaneP[0]) + aimLaneP[1]
            else:
                self.aP[0] = aimLaneP[0] + math.sin(theta) * (-self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.cos(theta) * (-self.LorR) * roadWidth / 2

    def my_ap_way(self,landetctect_index,pixelY,pixelX,roadWidth):
        a2, a1, a0 = np.polyfit(pixelY, pixelX, 2)  # 采用二次多项式拟合非零像素点（车道线）， 可画出此多项式曲线，之后看看怎么显示在当前帧上

        frontDistance = np.argsort(pixelY)[int(len(pixelY) / 8)]  # 这步是为了得到aimLanP，设定目标点距离的参数，这个参数对于车辆行走有影响，需要对应现场调到合适的参数
        aimLaneP = [pixelX[frontDistance], pixelY[frontDistance]]
        # 计算aimLaneP处斜率，从而得到目标点的像素坐标
        lane_Pkx = 2 * a2 * aimLaneP[0] + a1
        # lane_Pky = 2 * a2 * aimLaneP[1] + a1
        print('a2',a2)
        print('640-landetctect_index',640-landetctect_index)
        k_ver = - 1 / lane_Pkx
        print('k_ver',k_ver)
        # k_ver = - 1 / lane_Pkx
        theta = abs(math.atan(k_ver))  # 利用法线斜率求aP点坐标
        #右边的
        if int(640-landetctect_index) < 0:
            if lane_Pkx > 0 :
                self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
                if int(640-landetctect_index) < -50:
                    self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
                    self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
            else:
                self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.sin(theta) * (-self.LorR) * roadWidth / 2
                if int(640-landetctect_index) < -50:
                    self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
                    self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
        if int(640-landetctect_index) > 0:
            if lane_Pkx > 0 :
                self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.sin(theta) * (-self.LorR) * roadWidth / 2
                if int(640-landetctect_index) < 50:
                    self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
                    self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
            else:
                self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
                self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
                if int(640-landetctect_index) < 50:
                    self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
                    self.aP[1] = aimLaneP[1] + math.sin(theta) * (-self.LorR) * roadWidth / 2



        # if int(640-landetctect_index) > 0:
        #     # 逆时针外线
        #     if a2 >0 :
        #         if k_ver > 0:
        #             self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
        #             self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
        #         #可以不要
        #         else:
        #             self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
        #             self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
        # else:
        #     # 顺时针内线
        #     if int(640 - landetctect_index) < 0:
        #         if k_ver > 0:
        #             self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
        #             self.aP[1] = aimLaneP[1] + math.sin(theta) * (- self.LorR) * roadWidth / 2
        #         else:
        #             self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
        #             self.aP[1] = aimLaneP[1] + math.sin(theta) * (-self.LorR) * roadWidth / 2
        #     # 逆时针内线
        #     if int(640 - landetctect_index) > 0:
        #         if k_ver > 0:
        #             self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
        #             self.aP[1] = aimLaneP[1] + math.sin(theta) * (-self.LorR) * roadWidth / 2
        #         else:
        #             self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
        #             self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2
    def my_ap_way1(self,landetctect_index,pixelY,pixelX,roadWidth): #先判断左右 在顺逆
        a2, a1, a0 = np.polyfit(pixelY, pixelX, 2)  # 采用二次多项式拟合非零像素点（车道线）， 可画出此多项式曲线，之后看看怎么显示在当前帧上

        frontDistance = np.argsort(pixelY)[int(len(pixelY) / 8)]  # 这步是为了得到aimLanP，设定目标点距离的参数，这个参数对于车辆行走有影响，需要对应现场调到合适的参数
        aimLaneP = [pixelX[frontDistance], pixelY[frontDistance]]
        # 计算aimLaneP处斜率，从而得到目标点的像素坐标
        lane_Pkx = 2 * a2 * aimLaneP[0] + a1
        k_ver = - 1 / lane_Pkx
        theta = abs(math.atan(k_ver))  # 利用法线斜率求aP点坐标

        #ni 外线

        if a2>0:
            #顺时针外线
                if k_ver >0:
                    #逆时针
                    self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
                    self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2

                else:
                    #顺时针
                    self.aP[0] = aimLaneP[0] + math.cos(theta) * (-self.LorR) * roadWidth / 2
                    self.aP[1] = aimLaneP[1] + math.sin(theta) * (-self.LorR) * roadWidth / 2

        if a2<0:
                if k_ver >0:
                    self.aP[0] = aimLaneP[0] + math.cos(theta) * (self.LorR) * roadWidth / 2
                    self.aP[1] = aimLaneP[1] + math.sin(theta) * (self.LorR) * roadWidth / 2







    def spin(self):
        print(self.cap.isOpened())
        ret, img = self.cap.read()
        # print(ret)
        #img = cv2.resize(img, (1280, 720))
        global lanePk
        if ret == True:
            # 图像预处理
            # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # kernel = np.ones((3, 3), np.uint8)
            # gray_img = cv2.erode(gray_img, kernel, iterations=1)
            cv2.imshow('img',img)
            mpv_channel = utils.computer_img_mean(img)
            if mpv_channel >= 50.0:
                gray_img = utils.thresholding(img)
            else:
                gray_img = pipeline.thresholding(img)
            #iteration的值越高，模糊程度(腐蚀程度)就越高 呈正相关关系
            #图像腐蚀 加上高斯模糊 就可以使得图像的色彩更加突出
            #可以是色彩追踪更加精准，少了很多的颜色干扰
            # gray_Blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # # gray_Blur = np.power(gray_Blur / 255.0, 1.5)
            # retval, dst = cv2.threshold(gray_Blur, 0, 255, cv2.THRESH_OTSU)
            # dst = cv2.dilate(dst, None, iterations=2)
            # origin_thr = cv2.erode(dst, None, iterations=6)

            origin_thr = np.zeros_like(gray_img)
            # origin_thr[(gray_img >= 150)] = 255
            origin_thr[(gray_img >= 1)] = 255

            # 每一帧原始二值图都进行一次透视变换，这里透视变换矩阵M会影响性能
            # binary_warped = origin_thr
            binary_warped = cv2.warpPerspective(origin_thr, self.M, (1280, 720))

            # 二值图合并为三通道
            show_img = cv2.merge((binary_warped, binary_warped, binary_warped))

            histogram_x = np.sum(binary_warped[int(binary_warped.shape[0] * 9 / 10):, :], axis=0)  # 直方图车道线检测，只检测下半部分图
            lane_base = np.argmax(histogram_x)  # 取最大值作为车道线起始点
            midpoint_x = int(histogram_x.shape[0] / 2)  # 车道线中点，没有车道线的话就取中点



            histogram_y = np.sum(binary_warped[0:binary_warped.shape[0], :], axis=1)  # y轴上下直方图检测
            histogramy = np.sum(histogram_y[0:720])
            # midpoint_y = 320  # int(histogram.shape[0]/2)   # 这个320是不是也是人为可调的？

            # ax1.plot(np.arange(0, 720, 1), histogram_y)
            # plt.show()
            # plt.pause(0.0000001)
            # plt.close()

            # upper_half_histSum = np.sum(histogram_y[0:midpoint_y])
            # lower_half_histSum = np.sum(histogram_y[midpoint_y:])
            # try:
            #     hist_sum_y_ratio = (upper_half_histSum) / (lower_half_histSum)  # 防止除数为0异常，hist_sum_y_ratio参数有什么用？看下面貌似是用来检测左右车道线的
            # except:
            #     hist_sum_y_ratio = 1
            # print(hist_sum_y_ratio)

            nwindows = 15
            window_height = int(binary_warped.shape[0] / nwindows)  # 定义10个车道线检测框
            nonzero = binary_warped.nonzero()   # 获取非零像素对应坐标
            nonzeroy = np.array(nonzero[0])  # 获取非零的y坐标
            nonzerox = np.array(nonzero[1])  # 获取非零的x坐标
            lane_current = lane_base    # 当前车道线起点
            margin = 200    # 检测框大小，可调
            minpix = 25     # 检测框是否有效的阈值

            lane_inds = []
            # 滑动窗口
            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_x_low = lane_current - margin
                win_x_high = lane_current + margin
                cv2.rectangle(show_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 3)
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                        nonzerox < win_x_high)).nonzero()[0]    # 筛选好的框，在目标框内有y的非零像素则视为有效框
                lane_inds.append(good_inds)
                if len(good_inds) > minpix:  # 大于此阈值的最小像素数量，则更新当前车道线位置
                    lane_current = int(np.mean(nonzerox[good_inds]))  # 取非零像素均值作为新的车道线位置
                elif window >= 1:  # 如果当前帧超过3个框都没有满足上述阈值，则说明当前帧没有车道线，可调参数
                    # self.self.noLanePub.publish(0)
                    # self.self.noLanePub.publish(0)
                    break


            lane_inds = np.concatenate(lane_inds)   # 把所有框连在一起

            # 找到所有的车道线非零像素点
            pixelX = nonzerox[lane_inds]
            pixelY = nonzeroy[lane_inds]

            # calculate the aimPoint
            if (pixelX.size == 0):
                return
            frontDistance = np.argsort(pixelY)[
                int(len(pixelY) / 8)]  # 这步是为了得到aimLanP，设定目标点距离的参数，这个参数对于车辆行走有影响，需要对应现场调到合适的参数
            aimLaneP = [pixelX[frontDistance], pixelY[frontDistance]]
            self.my_ap_way(lane_base,pixelX,pixelY,roadWidth)
            print('self.aP',self.aP)
            self.line_ap.update(self.aP )

            # print('ap_line.detected',ap_line.detected)
            if self.line_ap.detected == False:
                Ap = self.line_ap.best_ap
                self.aP[0] = Ap[0]
                self.aP[1] = Ap[1]
                print('ap02', Ap)
            # 拟合的曲线形式：y = a2*x^2+a1*x+a0




                # lanePk = lane_Pky

            # # 官方左右车道线判别算法
            #lanePk = 2 * a2 * aimLaneP[0] + a1

            # if (abs(lanePk) < 0.1):  # 斜率判断参数是否可调？
            #     if lane_base >= midpoint_x:  # 如果车道线在右半边
            #         LorR = -1.25    # 左右车道线因子，估计是根据路宽和斜率综合估算的，防止aP点落到车道外边
            #     else:
            #         if hist_sum_y_ratio < 0.1:
            #             LorR = -1.25
            #         else:
            #             LorR = 0.8
            #     self.aP[0] = aimLaneP[0] + LorR * roadWidth / 2
            #     self.aP[1] = aimLaneP[1]
            # else:
            #     if (2 * a2 * aveX + a1) > 0:  # 斜率大于0
            #         if a2 > 0:
            #             # x_intertcept = (-a1+(abs(a1*a1-4*a2*(a0 - 1280))**0.5))/(2*a2)
            #             x_intertcept = (-a1 + (abs(a1 * a1 - 4 * a2 * (a0 - 1099.0)) ** 0.5)) / (2 * a2)  # 求截距
            #
            #         else:
            #             x_intertcept = (-a1 - (abs(a1 * a1 - 4 * a2 * (a0 - 1099.0)) ** 0.5)) / (2 * a2)
            #
            #
            #     else:  # 斜率小于0
            #         if a2 > 0:
            #             # x_intertcept = (-a1-(abs(a1*a1-4*a2*(a0 - 1280))**0.5))/(2*a2)
            #             x_intertcept = (-a1 - (abs(a1 * a1 - 4 * a2 * (a0 - 1099.0)) ** 0.5)) / (2 * a2)
            #
            #         else:
            #             x_intertcept = (-a1 + (abs(a1 * a1 - 4 * a2 * (a0 - 1099.0)) ** 0.5)) / (2 * a2)
            #
            #     if (x_intertcept > 1200):  # 默认599
            #         # LorR = -1.4  # RightLane
            #         LorR = -1.4
            #     else:
            #         # LorR = 0.8  # LeftLane
            #         LorR = 0.8
            #
            #     k_ver = - 1 / lanePk
            #     theta = math.atan(k_ver)  # 利用法线斜率求aP点坐标
            #     self.aP[0] = aimLaneP[0] + math.cos(theta) * (LorR) * roadWidth / 2
            #     self.aP[1] = aimLaneP[1] + math.sin(theta) * (LorR) * roadWidth / 2

            # 拟合曲线的可视化
            # a3 = [a2, a1, a0]
            # p1 = np.poly1d(a3)
            # yvals = p1(pixelY)
            # plt.ion()
            # plt.clf()
            #
            #
            # plt.subplot(3, 1, 1)
            # plt.plot(np.arange(0, 1280, 1), histogram_x)
            # plt.subplot(3, 1, 2)
            # plt.plot(np.arange(0, 720, 1), histogram_y)
            # plt.subplot(3, 1, 3)
            #
            #
            # plt.plot(yvals, pixelY, 'r', label='polyfit values')
            # plt.xlabel('x axis')
            # plt.ylabel('y axis')
            # plt.imshow(binary_warped)
            # plt.scatter(aimLaneP[0], aimLaneP[1], color='b', s=200)
            # plt.scatter(self.aP[0], self.aP[1], color='r', s=200)
            # plt.pause(0.001)
            # plt.ioff()


            # 逆时针走出盲区如果看到左车道线，让它直行，直到看到右车道线
            if lane_base < 400:
                if self.aP[0] < aimLaneP[0]:
                    self.aP[0] = midpoint_x
                    self.aP[1] = aimLaneP[1]
                else:
                    pass
            # 顺时针走出盲区如果看到右车道线，让它直行，直到看到左车道线
            if lane_base > 900:
                if self.aP[0] > aimLaneP[0]:
                    self.aP[0] = midpoint_x
                    self.aP[1] = aimLaneP[1]

            # 激光雷达衔接
            if aimLaneP[1] > 500:
                if len(self.lanpk_before) > 20:
                    print(len(self.lanpk_before))
                    if self.lanpk_before[-20] > 0:
                        self.aP[0] = 200
                        self.aP[1] = 600
                        # time.sleep(0.1)
                        # self.aP[0] = midpoint_x
                    else:
                        self.aP[0] = 1000
                        self.aP[1] = 600

                # time.sleep(1)
                # self.aP[0] = midpoint_x

            # # 人行横道判别
            # if histogramy > 50000000:
            #     if lanePk > 0:
            #         self.aP[0] = 400
            #         self.aP[1] = aimLaneP[1]


            # 把aP点可视化
            # 把aimLaneP点可视化
            cv2.circle(show_img, (aimLaneP[0], aimLaneP[1]), 40, (255, 0, 0), -1)
            # plt.scatter(aimLaneP[0], aimLaneP[1], color='b', s=300)
            cv2.circle(show_img, (int(self.aP[0]), int(self.aP[1])), 40, (0, 0, 255), -1)

            # 计算目标点的真实坐标，这里x_cmPerPixel\y_cmPerPixel虽然官方说他们做了几次测试说没必要标定了，但是有条件的可以作🌿一些修正
            # 这里的映射关系还没完全弄懂，到时候可实地勘测一下具体的真实距离关系，再看看有没有必要调整一些参数和偏移量
            self.aP[0] = (self.aP[0] - 599) * x_cmPerPixel
            self.aP[1] = (680 - self.aP[1]) * y_cmPerPixel + y_offset

            # 排除人行道的干扰，这里的算法能否作进一步考虑？
            # if (self.lastP[0] > 0.001 and self.lastP[1] > 0.001):
            #     if (((self.aP[0] - self.lastP[0]) ** 2 + (  # 这里貌似是通过计算和上一次目标点的距离，综合计时器来判断人行道
            #             self.aP[1] - self.lastP[1]) ** 2 > 2500) and self.Timer < 2):  # To avoid the mislead by walkers
            #         self.aP = self.lastP[:]
            #         self.Timer += 1
            #     else:
            #         self.Timer = 0
            #
            # self.lastP = self.aP[:]  # 记录上一个目标点

            # 根据pure persuit算法计算实际转角，这里会根据自行车模型作一些修正
            steerAngle = math.atan(2 * I * self.aP[0] / (self.aP[0] * self.aP[0] + (self.aP[1] + D) * (self.aP[1] + D)))

            # self.cam_cmd.angular.z = k * steerAngle
            self.angular_z = k * steerAngle
            print("steerAngle=", steerAngle)
            # self.cmdPub.publish(self.cam_cmd)
            print("cam_cmd.angular.z=", self.angular_z)
            # self.imagePub.publish(self.cvb.cv2_to_imgmsg(binary_warped))  # binary_warped
            #
            # cv2.imshow('origin', img)
            # text_angle = "topic: cam_cmd.angular.z= " + str(self.angular_z) + " " + "topic: steerAngle=" + str(steerAngle)
            # cv2.putText(show_img, text_angle, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow('binary_warped', show_img)
            cv2.waitKey()


if __name__ == '__main__':
    # rospy.init_node('lane_vel', anonymous=True)
    # rate = rospy.Rate(10)

    cam = camera()
    # print(rospy.is_shutdown())  # FALSE
    while True:
        cam.spin()
        print('betweeen == cam.spin ==')
            #rate.sleep()
        c = cv2.waitKey(50)
        if c == 27:  # 按esc退出
            break
    # except rospy.ROSInterruptException:
    #     print(rospy.ROSInterruptException)
    #     pass
    cv2.destroyAllWindows()
