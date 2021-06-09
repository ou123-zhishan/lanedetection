import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
# import utils
import utils_V2 as utils
# import pipeline
import math
import glob
import line
# import rospy
import aimpoin_line
plt.style.use('ggplot')
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge

# 距离映射
x_cmPerPixel = 90 / 665.00#  ？？？？？？？？
y_cmPerPixel = 81 / 680.00
roadWidth = 665

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

        self.cap = cv2.VideoCapture(r'Video_1.avi')
        #ROS
        # self.cap = cv2.VideoCapture('/dev/video0')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)#摄像头分辨率1280*720
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #ROS
        # self.imagePub = rospy.Publisher('images', Image, queue_size=1)
        # self.cmdPub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # self.cam_cmd = Twist()
        # self.cvb = CvBridge()
        # 透视变换
        # src_points = np.array([[11, 558], [225, 431], [994, 434], [1270, 560]], dtype="float32")  # 源图像中待测矩形的四点坐标
        # dst_points = np.array([[87., 699.], [23., 267.], [1105., 216.], [1063., 704.]], dtype="float32")  # 目标图像中矩形的四点坐标

        self.left_line = line.Line()
        self.right_line = line.Line()
        self.ap_line = aimpoin_line.Aimpoin_line()
        self.cal_imgs = utils.get_images_by_dir('camera_cal')

        self.M, self.Minv = utils.get_M_Minv()
        self.object_points, self.img_points = utils.calibrate(self.cal_imgs, grid=(9, 6))

        self.ap = [0.0, 0.0]
        self.lastP = [0.0, 0.0]
        self.Timer = 0
        self.lanpk_before = []
        self.count = 0

        self.LorR = 0.8 #aimPoint和aP距离偏移量

    def __del__(self):
        self.cap.release()  # 释放摄像头  关闭摄像头
        # self.out.release()

    def spin(self):
        print(self.cap.isOpened())
        ret, img = self.cap.read()
        global lanePk
        if ret == True :
            mpv_channel = utils.computer_img_mean(img)
            if mpv_channel >= 70:
                gray_img = utils.thresholding(img)
            else:
                gray_img = utils.thresholding_black(img)
            origin_thr = np.zeros_like(gray_img)
            origin_thr[(gray_img >= 1)] = 255
            cv2.imshow('origin_thr', origin_thr)
            binary_warped = cv2.warpPerspective(origin_thr, self.M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            cv2.imshow('binary',binary_warped)
            if (self.left_line.detected ):

                left_fit, left_lane_inds,leftx,lefty,flag,landetctect_index = utils.find_line_by_previous_max(binary_warped,self.left_line.current_fit)
                print('left_fit0', left_fit)
                print('leftx,lefty0',leftx,lefty)
            else:
                # 拟合车道线
                left_fit,left_lane_inds ,show_img,flag,leftx,lefty,landetctect_index = utils.find_line_max(binary_warped ,nwindows=15, margin=200, minpix=25)
                # if flag == 1:
                print('left_fit1', left_fit)
                print('leftx,lefty1', leftx, lefty)
                cv2.imshow('new', show_img)
                cv2.waitKey(1)
                # cv2.waitKey()
                # self.left_line.update(left_fit)
                #      how_img)

            # if left_fit!=[]:
            #    self.left_line.update(left_fit)
            if flag == 1:
                self.left_line.update(left_fit)
                print('left_fit3',left_fit)
                print('leftx,lefty3', leftx, lefty)
                result, out_img1, out_img0,angular_z  = utils.draw_area_max(img, binary_warped, self.Minv, left_fit,leftx,lefty,landetctect_index,self.ap_line)
                # color_warp, out_img, left_fit, right_fit, ploty = utils.fit_polynomial(img, Minv, thresholding_wrape0,
                #曲率
                #ROS
                #self.cam_cmd.angular.z = angular_z
                # self.cam_cmd.linear.x = 0.12
                # print("steerAngle=", angular_z/(-10))
                # self.cmdPub.publish(self.cam_cmd)
                # print("cam_cmd.angular.z=", angular_z)
                # self.imagePub.publish(self.cvb.cv2_to_imgmsg(binary_warped))
                curvature = utils.calculate_curv_and_pos_max(binary_warped, left_fit)
                imge_ret = utils.draw_values_max( result, curvature)
                # cv2.imshow('out', out_img1)
                # # # cv2.imshow('re',imge_ret)
                cv2.imshow('clo',out_img0)
                cv2.imshow('re', imge_ret)
                # cv2.imshow('new', show_img)
                # cv2.waitKey()
                # self.left_line.update(left_fit)
                #      how_img)
                cv2.waitKey(1)
            #ROS
            # self.cam_cmd.linear.x = 0.12
            # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
            # if (self.left_line.detected and self.right_line.detected) or (self.left_line.detected == False and self.right_line.detected) \
            #         or (self.left_line.detected and self.right_line.detected == False):
            #     left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(binary_warped ,
            #                                                                                        self.left_line.current_fit,
            #                                                                                        self.right_line.current_fit)
            # else:
            #     # 拟合车道线
            #     left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = utils.find_line(binary_warped,
            #                                                                                 nwindows=9,
            #                                                                                     margin=100, minpix=50)
            #     # cv2.imshow('out', out_img)
            #     # color_warp, out_img, left_fit, right_fit, ploty = utils.fit_polynomial(img, Minv, thresholding_wrape0,
            #     #                                                                        nwindows=9, margin=100, minpix=50)
            # self.left_line.update(left_fit)
            # self.right_line.update(right_fit)
            # # # left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholding_wrape,nwindows=9, margin=100, minpix=50 )
            # result, out_img1, out_img0 = utils.draw_area(img, binary_warped, self.Minv, left_fit, right_fit)
            # curvature, distance_from_center = utils.calculate_curv_and_pos(binary_warped, left_fit,right_fit )
            # imge_ret = utils.draw_values_max( result, curvature)
            # cv2.imshow('re', imge_ret)
            #                                                                      # nwindows=9, margin=100, minpix=50)
            # cv2.imshow('out',out_img0  )
            # # # cv2.imshow('re',imge_ret)
            # cv2.imshow('clo',out_img1)
            # cv2.imshow('new', show_img )
            # # cv2.waitKey()
            # # self.left_line.update(left_fit)
            # #      how_img)
            #  cv2.waitKey()

if __name__ == '__main__':
    #ros
    # rospy.init_node('lane_vel', anonymous=True)
    # rate = rospy.Rate(10)
    cam = camera()
    while True:
        cam.spin()
        print('betweeen == cam.spin ==')
        c = cv2.waitKey(5)
        if c == 5:
            break
    cv2.destroyAllWindows()





