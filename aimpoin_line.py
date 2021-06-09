import numpy as np
class Aimpoin_line():
    def __init__(self):
        #检测是否在误差范围内
        self.detected = False
        self.recent_ap = [np.array([False])] #[np.array([False])] 与 [] python2.7识别不出来前一个 只能[] python3.6都可以 会自动填维度
        self.bestx = None
        self.best_ap = None
        self.current_ap =[np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs_ap = np.array([0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
    #X Y 的允许误差
    def check_detected_ap(self):
        if (self.diffs_ap[0] < 300 and self.diffs_ap[1] < 300 ):
            return True
        else:
            return False
    def update(self,fit_ap):
        if fit_ap is not None:
            fit_ap0 = np.array(fit_ap)
            if self.best_ap is not None:
                self.diffs_ap = abs(fit_ap0 - self.best_ap)
                print('self.diffs_ap',self.diffs_ap)
                if self.check_detected_ap():
                    self.detected = True
                    # 取前十帧的均值作为下一帧的输出
                    if len(self.recent_ap) >10:
                        self.recent_ap = self.recent_ap[int(len(self.recent_ap)-10):]
                        self.recent_ap.append(fit_ap0)
                    else:
                        self.recent_ap.append(fit_ap0)
                    self.best_ap = np.average(self.recent_ap, axis=0)
                    print('self.best_ap',self.best_ap,type(self.best_ap),len(self.best_ap),type(self.recent_ap),self.recent_ap)
                    self.current_ap = fit_ap0
                else:
                    self.detected = False
            else:
                print('fit_ap0',fit_ap0)
                fit_ap0 = np.array(fit_ap)
                self.best_ap = fit_ap0
                #print('self.best_ap',self.best_ap)
                self.current_ap = fit_ap0
                self.detected = True
                self.recent_ap.append(fit_ap0)






