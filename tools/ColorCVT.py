from PyQt5.QtWebEngineWidgets import *
from PyQt5 import QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2 as cv
from PyQt5.QtWidgets import *
from ColorCVT_UI import *
import numpy as np
import sys

# 原图数据
img_original = None
# 判断radiobutton的标志位
hsv_flag = False
hls_flag = False
sobelx_flag = False
mag_flag = False
dir_flag = False
lab_flag = False
luv_flag = False
# 阈值
hsv_min = np.array([0, 0, 0])
hsv_max = np.array([255, 255, 255])
hls_min = np.array([0, 0, 0])
hls_max = np.array([255, 255, 255])
sobelx_thresh = np.array([0, 255])
mag_thresh = np.array([0, 255])
dir_thresh = np.array([0, np.pi / 2])
lab_min = np.array([0, 0, 0])
lab_max = np.array([255, 255, 255])
luv_min = np.array([0, 0, 0])
luv_max = np.array([255, 255, 255])


class MyWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # 界面初始化以及事件绑定
        self.img_select_pushButton.clicked.connect(self.select_img)
        self.hsv_radioButton.toggled.connect(self.hsv_img)
        self.hls_radioButton.toggled.connect(self.hls_img)
        self.sobelx_radioButton.toggled.connect(self.sobelx_img)
        self.mag_radioButton.toggled.connect(self.mag_img)
        self.dir_radioButton.toggled.connect(self.dir_img)
        self.lab_radioButton.toggled.connect(self.lab_img)
        self.luv_radioButton.toggled.connect(self.luv_img)
        self.hsv_radioButton.setEnabled(False)
        self.hls_radioButton.setEnabled(False)
        self.sobelx_radioButton.setEnabled(False)
        self.mag_radioButton.setEnabled(False)
        self.dir_radioButton.setEnabled(False)
        self.lab_radioButton.setEnabled(False)
        self.luv_radioButton.setEnabled(False)
        self.c1_l_horizontalSlider.setEnabled(False)
        self.c1_h_horizontalSlider.setEnabled(False)
        self.c2_l_horizontalSlider.setEnabled(False)
        self.c2_h_horizontalSlider.setEnabled(False)
        self.c3_l_horizontalSlider.setEnabled(False)
        self.c3_h_horizontalSlider.setEnabled(False)
        self.slider_init()
        self.c1_l_horizontalSlider.valueChanged.connect(self.c1_l_valuechange)
        self.c1_h_horizontalSlider.valueChanged.connect(self.c1_h_valuechange)
        self.c2_l_horizontalSlider.valueChanged.connect(self.c2_l_valuechange)
        self.c2_h_horizontalSlider.valueChanged.connect(self.c2_h_valuechange)
        self.c3_l_horizontalSlider.valueChanged.connect(self.c3_l_valuechange)
        self.c3_h_horizontalSlider.valueChanged.connect(self.c3_h_valuechange)

    # 尝试调用cv.inRange方法但是在pyqt5下一直出现崩溃情况，无奈根据inRange原理重写其实现
    # 原理其实是将每个通道内介于阈值之间的像素值赋值255（关于等于阈值的情况我没有深究，仍然将其赋值255），其余赋值为0
    # QImage模块没有HSV以及HLS格式，所以只能调用OpenCV的imshow方法，原本是想显示在img_label上的...
    def update_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        # HSV
        if hsv_flag is True:
            hsv_img = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)
            h = hsv_img[:, :, 0]
            h_bin = np.zeros_like(h)
            h_bin[(h >= hsv_min[0]) & (h <= hsv_max[0])] = 255

            s = hsv_img[:, :, 1]
            s_bin = np.zeros_like(s)
            s_bin[(s >= hsv_min[1]) & (s <= hsv_max[1])] = 255

            v = hsv_img[:, :, 2]
            v_bin = np.zeros_like(v)
            v_bin[(v >= hsv_min[2]) & (v <= hsv_max[2])] = 255

            # 二值化只需要一个通道，这里随便取其中一个通道
            hsv_bin = np.zeros_like(h)
            hsv_bin[(h_bin == 255) & (s_bin == 255) & (v_bin == 255)] = 255
            cv.imshow('HSV_img', hsv_bin)
        # HLS
        elif hls_flag is True:
            hls_img = cv.cvtColor(img_original, cv.COLOR_BGR2HLS)
            h = hls_img[:, :, 0]
            h_bin = np.zeros_like(h)
            h_bin[(h >= hls_min[0]) & (h <= hls_max[0])] = 255

            l = hls_img[:, :, 1]
            l_bin = np.zeros_like(l)
            l_bin[(l >= hls_min[1]) & (l <= hls_max[1])] = 255

            s = hls_img[:, :, 2]
            s_bin = np.zeros_like(s)
            s_bin[(s >= hls_min[2]) & (s <= hls_max[2])] = 255

            # 二值化只需要一个通道，这里随便取其中一个通道
            hls_bin = np.zeros_like(h)
            hls_bin[(h_bin == 255) & (l_bin == 255) & (s_bin == 255)] = 255
            cv.imshow('HLS_img', hls_bin)
        # sobelx
        elif sobelx_flag is True:
            gray_img = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
            abs_sobelx = np.absolute(cv.Sobel(gray_img, cv.CV_64F, 1, 0))  # dx=1 dy=0，采用默认卷积核大小
            scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))  # 限制值在0~255
            # 二值化只需要一个通道，这里随便取其中一个通道
            sobelx_img = np.zeros_like(scaled_sobelx)
            sobelx_img[(scaled_sobelx >= sobelx_thresh[0]) & (scaled_sobelx <= sobelx_thresh[1])] = 255
            cv.imshow('sobelx_img', sobelx_img)
        # mag
        elif mag_flag is True:
            gray_img = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
            sobelx = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)  # 卷积核3*3
            sobely = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)
            gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
            scale_factor = np.max(gradmag) / 255
            gradmag = (gradmag / scale_factor).astype(np.uint8)
            mag_img = np.zeros_like(gradmag)
            mag_img[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255
            cv.imshow('mag_img', mag_img)
        # dir
        elif dir_flag is True:
            gray_img = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
            sobelx = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)  # 卷积核3*3
            sobely = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)
            absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
            dir_img = np.zeros_like(absgraddir)
            dir_img[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 255
            cv.imshow('dir_img', dir_img)
        # lab
        elif lab_flag is True:
            lab_img = cv.cvtColor(img_original, cv.COLOR_BGR2Lab)  # 有个疑问，不知道BGR2Lab和BGR2LAB有没有什么区别
            l = lab_img[:, :, 0]
            l_bin = np.zeros_like(l)
            l_bin[(l >= lab_min[0]) & (l <= lab_max[0])] = 255

            a = lab_img[:, :, 1]
            a_bin = np.zeros_like(a)
            a_bin[(a >= lab_min[1]) & (a <= lab_max[1])] = 255

            b = lab_img[:, :, 2]
            b_bin = np.zeros_like(b)
            b_bin[(b >= lab_min[2]) & (b <= lab_max[2])] = 255

            # 二值化只需要一个通道，这里随便取其中一个通道
            lab_bin = np.zeros_like(l)
            lab_bin[(l_bin == 255) & (a_bin == 255) & (b_bin == 255)] = 255
            cv.imshow('lab_img', lab_bin)
        # luv
        elif luv_flag is True:
            luv_img = cv.cvtColor(img_original, cv.COLOR_BGR2Luv)  # 疑问同上，不知道BGR2Luv和BGR2LUV有没有什么区别
            l = luv_img[:, :, 0]
            l_bin = np.zeros_like(l)
            l_bin[(l >= luv_min[0]) & (l <= luv_max[0])] = 255

            u = luv_img[:, :, 1]
            u_bin = np.zeros_like(u)
            u_bin[(u >= luv_min[1]) & (u <= luv_max[1])] = 255

            v = luv_img[:, :, 2]
            v_bin = np.zeros_like(v)
            v_bin[(v >= luv_min[2]) & (v <= luv_max[2])] = 255

            # 二值化只需要一个通道，这里随便取其中一个通道
            luv_bin = np.zeros_like(l)
            luv_bin[(l_bin == 255) & (u_bin == 255) & (v_bin == 255)] = 255
            cv.imshow('luv_img', luv_bin)

    # slider的逻辑
    def c1_l_valuechange(self):
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        if hsv_flag is True:
            self.c1_l_label.setText('H min: ' + str(self.c1_l_horizontalSlider.value()))
            hsv_min[0] = np.int(self.c1_l_horizontalSlider.value())
            self.textEdit.setText('hsv_min=' + str(hsv_min) + '\n' + 'hsv_max=' + str(hsv_max))
            self.update_img()
        elif hls_flag is True:
            self.c1_l_label.setText('H min: ' + str(self.c1_l_horizontalSlider.value()))
            hls_min[0] = np.int(self.c1_l_horizontalSlider.value())
            self.textEdit.setText('hls_min=' + str(hls_min) + '\n' + 'hls_max=' + str(hls_max))
            self.update_img()
        elif lab_flag is True:
            self.c1_l_label.setText('L min: ' + str(self.c1_l_horizontalSlider.value()))
            lab_min[0] = np.int(self.c1_l_horizontalSlider.value())
            self.textEdit.setText('lab_min=' + str(lab_min) + '\n' + 'lab_max=' + str(lab_max))
            self.update_img()
        elif luv_flag is True:
            self.c1_l_label.setText('L min: ' + str(self.c1_l_horizontalSlider.value()))
            luv_min[0] = np.int(self.c1_l_horizontalSlider.value())
            self.textEdit.setText('luv_min=' + str(luv_min) + '\n' + 'luv_max=' + str(luv_max))
            self.update_img()
        elif sobelx_flag is True:
            self.c1_l_label.setText('min: ' + str(self.c1_l_horizontalSlider.value()))
            sobelx_thresh[0] = np.int(self.c1_l_horizontalSlider.value())
            self.textEdit.setText('sobelx_min=' + str(sobelx_thresh[0]) + '\n' + 'sobelx_max=' + str(sobelx_thresh[1]))
            self.update_img()
        elif mag_flag is True:
            self.c1_l_label.setText('min: ' + str(self.c1_l_horizontalSlider.value()))
            mag_thresh[0] = np.int(self.c1_l_horizontalSlider.value())
            self.textEdit.setText('mag_min=' + str(mag_thresh[0]) + '\n' + 'mag_max=' + str(mag_thresh[1]))
            self.update_img()
        elif dir_flag is True:
            self.c1_l_label.setText('min: ' + str(self.c1_l_horizontalSlider.value()) + '°')
            dir_thresh[0] = np.deg2rad(np.int(self.c1_l_horizontalSlider.value()))  # 把角度转换为弧度
            self.textEdit.setText('dir_min=' + str(self.c1_l_horizontalSlider.value()) + '\n' + 'dir_max=' + str(
                self.c1_h_horizontalSlider.value()))  # 直接显示
            self.update_img()

    def c1_h_valuechange(self):
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        if hsv_flag is True:
            self.c1_h_label.setText('H max: ' + str(self.c1_h_horizontalSlider.value()))
            hsv_max[0] = np.int(self.c1_h_horizontalSlider.value())
            self.textEdit.setText('hsv_min=' + str(hsv_min) + '\n' + 'hsv_max=' + str(hsv_max))
            self.update_img()
        elif hls_flag is True:
            self.c1_h_label.setText('H max: ' + str(self.c1_h_horizontalSlider.value()))
            hls_max[0] = np.int(self.c1_h_horizontalSlider.value())
            self.textEdit.setText('hls_min=' + str(hls_min) + '\n' + 'hls_max=' + str(hls_max))
            self.update_img()
        elif lab_flag is True:
            self.c1_h_label.setText('L max: ' + str(self.c1_h_horizontalSlider.value()))
            lab_max[0] = np.int(self.c1_h_horizontalSlider.value())
            self.textEdit.setText('lab_min=' + str(lab_min) + '\n' + 'lab_max=' + str(lab_max))
            self.update_img()
        elif luv_flag is True:
            self.c1_h_label.setText('L max: ' + str(self.c1_h_horizontalSlider.value()))
            luv_max[0] = np.int(self.c1_h_horizontalSlider.value())
            self.textEdit.setText('luv_min=' + str(luv_min) + '\n' + 'luv_max=' + str(luv_max))
            self.update_img()
        elif sobelx_flag is True:
            self.c1_h_label.setText('max: ' + str(self.c1_h_horizontalSlider.value()))
            sobelx_thresh[1] = np.int(self.c1_h_horizontalSlider.value())
            self.textEdit.setText('sobelx_min=' + str(sobelx_thresh[0]) + '\n' + 'sobelx_max=' + str(sobelx_thresh[1]))
            self.update_img()
        elif mag_flag is True:
            self.c1_h_label.setText('max: ' + str(self.c1_h_horizontalSlider.value()))
            mag_thresh[1] = np.int(self.c1_h_horizontalSlider.value())
            self.textEdit.setText('mag_min=' + str(mag_thresh[0]) + '\n' + 'mag_max=' + str(mag_thresh[1]))
            self.update_img()
        elif dir_flag is True:
            self.c1_h_label.setText('max: ' + str(self.c1_h_horizontalSlider.value()) + '°')
            dir_thresh[1] = np.deg2rad(np.int(self.c1_h_horizontalSlider.value()))
            self.textEdit.setText('dir_min=' + str(self.c1_l_horizontalSlider.value()) + '\n' + 'dir_max=' + str(
                self.c1_h_horizontalSlider.value()))  # 直接显示
            self.update_img()

    def c2_l_valuechange(self):
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        if hsv_flag is True:
            self.c2_l_label.setText('S min: ' + str(self.c2_l_horizontalSlider.value()))
            hsv_min[1] = np.int(self.c2_l_horizontalSlider.value())
            self.textEdit.setText('hsv_min=' + str(hsv_min) + '\n' + 'hsv_max=' + str(hsv_max))
            self.update_img()
        elif hls_flag is True:
            self.c2_l_label.setText('L min: ' + str(self.c2_l_horizontalSlider.value()))
            hls_min[1] = np.int(self.c2_l_horizontalSlider.value())
            self.textEdit.setText('hls_min=' + str(hls_min) + '\n' + 'hls_max=' + str(hls_max))
            self.update_img()
        elif lab_flag is True:
            self.c2_l_label.setText('A min: ' + str(self.c2_l_horizontalSlider.value()))
            lab_min[1] = np.int(self.c2_l_horizontalSlider.value())
            self.textEdit.setText('lab_min=' + str(lab_min) + '\n' + 'lab_max=' + str(lab_max))
            self.update_img()
        elif luv_flag is True:
            self.c2_l_label.setText('U min: ' + str(self.c2_l_horizontalSlider.value()))
            luv_min[1] = np.int(self.c2_l_horizontalSlider.value())
            self.textEdit.setText('luv_min=' + str(luv_min) + '\n' + 'luv_max=' + str(luv_max))
            self.update_img()

    def c2_h_valuechange(self):
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        if hsv_flag is True:
            self.c2_h_label.setText('S max: ' + str(self.c2_h_horizontalSlider.value()))
            hsv_max[1] = np.int(self.c2_h_horizontalSlider.value())
            self.textEdit.setText('hsv_min=' + str(hsv_min) + '\n' + 'hsv_max=' + str(hsv_max))
            self.update_img()
        elif hls_flag is True:
            self.c2_h_label.setText('L max: ' + str(self.c2_h_horizontalSlider.value()))
            hls_max[1] = np.int(self.c2_h_horizontalSlider.value())
            self.textEdit.setText('hls_min=' + str(hls_min) + '\n' + 'hls_max=' + str(hls_max))
            self.update_img()
        elif lab_flag is True:
            self.c2_h_label.setText('A max: ' + str(self.c2_h_horizontalSlider.value()))
            lab_max[1] = np.int(self.c2_h_horizontalSlider.value())
            self.textEdit.setText('lab_min=' + str(lab_min) + '\n' + 'lab_max=' + str(lab_max))
            self.update_img()
        elif luv_flag is True:
            self.c2_h_label.setText('U max: ' + str(self.c2_h_horizontalSlider.value()))
            luv_max[1] = np.int(self.c2_h_horizontalSlider.value())
            self.textEdit.setText('luv_min=' + str(luv_min) + '\n' + 'luv_max=' + str(luv_max))
            self.update_img()

    def c3_l_valuechange(self):
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        if hsv_flag is True:
            self.c3_l_label.setText('V min: ' + str(self.c3_l_horizontalSlider.value()))
            hsv_min[2] = np.int(self.c3_l_horizontalSlider.value())
            self.textEdit.setText('hsv_min=' + str(hsv_min) + '\n' + 'hsv_max=' + str(hsv_max))
            self.update_img()
        elif hls_flag is True:
            self.c3_l_label.setText('S min: ' + str(self.c3_l_horizontalSlider.value()))
            hls_min[2] = np.int(self.c3_l_horizontalSlider.value())
            self.textEdit.setText('hls_min=' + str(hls_min) + '\n' + 'hls_max=' + str(hls_max))
            self.update_img()
        elif lab_flag is True:
            self.c3_l_label.setText('B min: ' + str(self.c3_l_horizontalSlider.value()))
            lab_min[2] = np.int(self.c3_l_horizontalSlider.value())
            self.textEdit.setText('lab_min=' + str(lab_min) + '\n' + 'lab_max=' + str(lab_max))
            self.update_img()
        elif luv_flag is True:
            self.c3_l_label.setText('V min: ' + str(self.c3_l_horizontalSlider.value()))
            luv_min[2] = np.int(self.c3_l_horizontalSlider.value())
            self.textEdit.setText('luv_min=' + str(luv_min) + '\n' + 'luv_max=' + str(luv_max))
            self.update_img()

    def c3_h_valuechange(self):
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        if hsv_flag is True:
            self.c3_h_label.setText('V max: ' + str(self.c3_h_horizontalSlider.value()))
            hsv_max[2] = np.int(self.c3_h_horizontalSlider.value())
            self.textEdit.setText('hsv_min=' + str(hsv_min) + '\n' + 'hsv_max=' + str(hsv_max))
            self.update_img()
        elif hls_flag is True:
            self.c3_h_label.setText('S max: ' + str(self.c3_h_horizontalSlider.value()))
            hls_max[2] = np.int(self.c3_h_horizontalSlider.value())
            self.textEdit.setText('hls_min=' + str(hls_min) + '\n' + 'hls_max=' + str(hls_max))
            self.update_img()
        elif lab_flag is True:
            self.c3_h_label.setText('B max: ' + str(self.c3_h_horizontalSlider.value()))
            lab_max[2] = np.int(self.c3_h_horizontalSlider.value())
            self.textEdit.setText('lab_min=' + str(lab_min) + '\n' + 'lab_max=' + str(lab_max))
            self.update_img()
        elif luv_flag is True:
            self.c3_h_label.setText('V max: ' + str(self.c3_h_horizontalSlider.value()))
            luv_max[2] = np.int(self.c3_h_horizontalSlider.value())
            self.textEdit.setText('luv_min=' + str(luv_min) + '\n' + 'luv_max=' + str(luv_max))
            self.update_img()

    # slider初始化
    def slider_init(self):
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        global hsv_min, hsv_max, hls_min, hls_max, sobelx_thresh, mag_thresh, dir_thresh, lab_min, lab_max, luv_min, luv_max
        hsv_min = np.array([0, 0, 0])
        hsv_max = np.array([255, 255, 255])
        hls_min = np.array([0, 0, 0])
        hls_max = np.array([255, 255, 255])
        sobelx_thresh = np.array([0, 255])
        mag_thresh = np.array([0, 255])
        dir_thresh = np.array([0, np.pi / 2])
        lab_min = np.array([0, 0, 0])
        lab_max = np.array([255, 255, 255])
        luv_min = np.array([0, 0, 0])
        luv_max = np.array([255, 255, 255])

        # dir阈值在0~2pi之间，化为角度方便调试
        if dir_flag is True:
            self.c1_l_horizontalSlider.setMinimum(0)
            self.c1_l_horizontalSlider.setMaximum(90)
            self.c1_h_horizontalSlider.setMinimum(0)
            self.c1_h_horizontalSlider.setMaximum(90)
        else:
            self.c1_l_horizontalSlider.setMinimum(0)
            self.c1_l_horizontalSlider.setMaximum(255)
            self.c1_h_horizontalSlider.setMinimum(0)
            self.c1_h_horizontalSlider.setMaximum(255)
        self.c2_l_horizontalSlider.setMinimum(0)
        self.c2_l_horizontalSlider.setMaximum(255)
        self.c2_h_horizontalSlider.setMinimum(0)
        self.c2_h_horizontalSlider.setMaximum(255)
        self.c3_l_horizontalSlider.setMinimum(0)
        self.c3_l_horizontalSlider.setMaximum(255)
        self.c3_h_horizontalSlider.setMinimum(0)
        self.c3_h_horizontalSlider.setMaximum(255)

        self.c1_l_horizontalSlider.setSingleStep(1)
        self.c1_l_horizontalSlider.setSingleStep(1)
        self.c1_h_horizontalSlider.setSingleStep(1)
        self.c1_h_horizontalSlider.setSingleStep(1)
        self.c2_l_horizontalSlider.setSingleStep(1)
        self.c2_l_horizontalSlider.setSingleStep(1)
        self.c2_h_horizontalSlider.setSingleStep(1)
        self.c2_h_horizontalSlider.setSingleStep(1)
        self.c3_l_horizontalSlider.setSingleStep(1)
        self.c3_l_horizontalSlider.setSingleStep(1)
        self.c3_h_horizontalSlider.setSingleStep(1)
        self.c3_h_horizontalSlider.setSingleStep(1)

        if dir_flag is True:
            self.c1_l_horizontalSlider.setValue(0)
            self.c1_h_horizontalSlider.setValue(90)
        else:
            self.c1_l_horizontalSlider.setValue(0)
            self.c1_h_horizontalSlider.setValue(255)
        self.c2_l_horizontalSlider.setValue(0)
        self.c2_h_horizontalSlider.setValue(255)
        self.c3_l_horizontalSlider.setValue(0)
        self.c3_h_horizontalSlider.setValue(255)

        self.c1_l_label.setText(str(self.c1_l_horizontalSlider.value()))
        self.c1_h_label.setText(str(self.c1_h_horizontalSlider.value()))
        self.c2_l_label.setText(str(self.c1_l_horizontalSlider.value()))
        self.c2_h_label.setText(str(self.c1_h_horizontalSlider.value()))
        self.c3_l_label.setText(str(self.c1_l_horizontalSlider.value()))
        self.c3_h_label.setText(str(self.c1_h_horizontalSlider.value()))

    # 显示图像在QT上，QImage支持显示的图像颜色空间格式不多...
    def display_img(self, img):
        rows, cols, channels = img.shape
        bytesPerLine = channels * cols
        qt_img = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        img = QtGui.QPixmap(qt_img).scaled(self.img_label.width(), self.img_label.height())
        self.img_label.setPixmap(img)
        self.img_label.setScaledContents(True)

    # 选择图片的逻辑
    def select_img(self):
        try:
            global img_original
            img_name, img_type = QFileDialog.getOpenFileName(self, '选择图片', '', '*.jpg;; *.png;;ALL files(*)')
            # 上一次打开过图片，这一次点击选择图片但是点取消的逻辑部分
            # print(img_name)
            if img_name == '':
                img_original = img_original
            else:
                img_original = cv.imread(img_name)
            img = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
            cv.destroyAllWindows()
            self.display_img(img)
            self.hsv_radioButton.setEnabled(True)
            self.hls_radioButton.setEnabled(True)
            self.sobelx_radioButton.setEnabled(True)
            self.mag_radioButton.setEnabled(True)
            self.dir_radioButton.setEnabled(True)
            self.lab_radioButton.setEnabled(True)
            self.luv_radioButton.setEnabled(True)
        except:
            return

    # radiobutton的逻辑
    def hsv_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        if img_original is None:
            return
        else:
            hsv_flag = True
            hls_flag = False
            sobelx_flag = False
            mag_flag = False
            dir_flag = False
            lab_flag = False
            luv_flag = False
            cv.destroyAllWindows()
            cv.namedWindow('HSV_img', cv.WINDOW_NORMAL)
            self.update_img()
        self.slider_init()
        self.c1_l_horizontalSlider.setEnabled(True)
        self.c1_h_horizontalSlider.setEnabled(True)
        self.c2_l_horizontalSlider.setEnabled(True)
        self.c2_h_horizontalSlider.setEnabled(True)
        self.c3_l_horizontalSlider.setEnabled(True)
        self.c3_h_horizontalSlider.setEnabled(True)

    def hls_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        if img_original is None:
            return
        else:
            hsv_flag = False
            hls_flag = True
            sobelx_flag = False
            mag_flag = False
            dir_flag = False
            lab_flag = False
            luv_flag = False
            cv.destroyAllWindows()
            cv.namedWindow('HLS_img', cv.WINDOW_NORMAL)
            self.update_img()
        self.slider_init()
        self.c1_l_horizontalSlider.setEnabled(True)
        self.c1_h_horizontalSlider.setEnabled(True)
        self.c2_l_horizontalSlider.setEnabled(True)
        self.c2_h_horizontalSlider.setEnabled(True)
        self.c3_l_horizontalSlider.setEnabled(True)
        self.c3_h_horizontalSlider.setEnabled(True)

    def sobelx_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        if img_original is None:
            return
        else:
            hsv_flag = False
            hls_flag = False
            sobelx_flag = True
            mag_flag = False
            dir_flag = False
            lab_flag = False
            luv_flag = False
            cv.destroyAllWindows()
            cv.namedWindow('sobelx_img', cv.WINDOW_NORMAL)
            self.update_img()
        self.slider_init()
        self.c1_l_horizontalSlider.setEnabled(True)
        self.c1_h_horizontalSlider.setEnabled(True)
        self.c2_l_horizontalSlider.setEnabled(False)
        self.c2_h_horizontalSlider.setEnabled(False)
        self.c3_l_horizontalSlider.setEnabled(False)
        self.c3_h_horizontalSlider.setEnabled(False)

    def mag_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        if img_original is None:
            return
        else:
            hsv_flag = False
            hls_flag = False
            sobelx_flag = False
            mag_flag = True
            dir_flag = False
            lab_flag = False
            luv_flag = False
            cv.destroyAllWindows()
            cv.namedWindow('mag_img', cv.WINDOW_NORMAL)
            self.update_img()
        self.slider_init()
        self.c1_l_horizontalSlider.setEnabled(True)
        self.c1_h_horizontalSlider.setEnabled(True)
        self.c2_l_horizontalSlider.setEnabled(False)
        self.c2_h_horizontalSlider.setEnabled(False)
        self.c3_l_horizontalSlider.setEnabled(False)
        self.c3_h_horizontalSlider.setEnabled(False)

    def dir_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        if img_original is None:
            return
        else:
            hsv_flag = False
            hls_flag = False
            sobelx_flag = False
            mag_flag = False
            dir_flag = True
            lab_flag = False
            luv_flag = False
            cv.destroyAllWindows()
            cv.namedWindow('dir_img', cv.WINDOW_NORMAL)
            self.update_img()
        self.slider_init()
        self.c1_l_horizontalSlider.setEnabled(True)
        self.c1_h_horizontalSlider.setEnabled(True)
        self.c2_l_horizontalSlider.setEnabled(False)
        self.c2_h_horizontalSlider.setEnabled(False)
        self.c3_l_horizontalSlider.setEnabled(False)
        self.c3_h_horizontalSlider.setEnabled(False)

    def lab_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        if img_original is None:
            return
        else:
            hsv_flag = False
            hls_flag = False
            sobelx_flag = False
            mag_flag = False
            dir_flag = False
            lab_flag = True
            luv_flag = False
            cv.destroyAllWindows()
            cv.namedWindow('lab_img', cv.WINDOW_NORMAL)
            self.update_img()
        self.slider_init()
        self.c1_l_horizontalSlider.setEnabled(True)
        self.c1_h_horizontalSlider.setEnabled(True)
        self.c2_l_horizontalSlider.setEnabled(True)
        self.c2_h_horizontalSlider.setEnabled(True)
        self.c3_l_horizontalSlider.setEnabled(True)
        self.c3_h_horizontalSlider.setEnabled(True)

    def luv_img(self):
        global img_original
        global hsv_flag, hls_flag, sobelx_flag, mag_flag, dir_flag, lab_flag, luv_flag
        if img_original is None:
            return
        else:
            hsv_flag = False
            hls_flag = False
            sobelx_flag = False
            mag_flag = False
            dir_flag = False
            lab_flag = False
            luv_flag = True
            cv.destroyAllWindows()
            cv.namedWindow('luv_img', cv.WINDOW_NORMAL)
            self.update_img()
        self.slider_init()
        self.c1_l_horizontalSlider.setEnabled(True)
        self.c1_h_horizontalSlider.setEnabled(True)
        self.c2_l_horizontalSlider.setEnabled(True)
        self.c2_h_horizontalSlider.setEnabled(True)
        self.c3_l_horizontalSlider.setEnabled(True)
        self.c3_h_horizontalSlider.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
    cv.destroyAllWindows()

    # HSV测试
    # img_original = cv.imread('D:/project/python_pro/graduation_project/video2frame/95.jpg')
    # hsv_img = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)
    # print(hsv_img.shape)
    # hsv_img = cv.inRange(hsv_img, hsv_min, hsv_max)
    # cv.namedWindow('HSV_img', cv.WINDOW_NORMAL)
    # cv.imshow('HSV_img', hsv_img)
    # cv.waitKey(0)

    # HLS测试
    # img_original = cv.imread('D:/project/python_pro/graduation_project/video2frame/95.jpg')
    # hsv_img = cv.cvtColor(img_original, cv.COLOR_BGR2HLS)
    # hsv_img = cv.inRange(hsv_img, hls_min, hls_max)
    # cv.namedWindow('HLS_img', cv.WINDOW_NORMAL)
    # cv.imshow('HLS_img', hsv_img)
    # cv.waitKey(0)
