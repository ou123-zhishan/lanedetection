# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:38:04 2017

@author: yang
"""
import numpy as np

# y
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_fitted = [np.array([False])]
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        # self.best_fit = None
        self.best_fit = []
        #polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
    
    def check_detected(self):
        if (self.diffs[0] < 0.01 and self.diffs[1] < 10.0 and self.diffs[2] < 1000.) and len(self.recent_fitted) > 0:
            return True
        else:
            return False
    
    #更新左右车道线left_fit ,right_fit
    def update(self,fit):
        if fit is not None:
            # if self.best_fit is not None:
            if self.best_fit != []:
                #上次与这次的偏差量 三通道
                #二次项拟合曲线 A2 A1 A0系数
                #当前的二次型系数 与上次保存前十次均值的二次型作对比
                self.diffs = abs(fit - self.best_fit)
                if self.check_detected():
                    #满足设定条件
                    self.detected =True
                    if len(self.recent_fitted)>10:
                        self.recent_fitted = self.recent_fitted[1:]
                        self.recent_fitted.append(fit)
                    else:
                        self.recent_fitted.append(fit)
                    self.best_fit = np.average(self.recent_fitted, axis=0)
                    print('self.recent_fitted', self.best_fit, type(self.best_fit), len(self.best_fit), type(self.recent_fitted),
                          self.recent_fitted)
                    self.current_fit = fit
                else:
                    self.detected = False
            #第一次有车道线
            else:
                self.best_fit = fit
                self.current_fit = fit
                #检测到车道线了
                self.detected=True
                #目前的车道线
                self.recent_fitted.append(fit)
            
        
        