import cv2
def Filter_images(self):
    video_input = 'offoutputrandom.avi'
    cap = cv2.VideoCapture(video_input)
    count = 1
    nx = 9
    ny = 6
    ret_, mtx, dist, rvecs, tvecs = self.getCameraCalibrationCoefficiebts(
        'camera_cal/calibration*.jpg', nx, ny)
    while (True):
        ret, image = cap.read()
        # print(ret)
        if ret:
            undistort_image = self.undistortImage(image, mtx, dist)
            cv2.imwrite('origin_image/' + str(count) + '.jpg', undistort_image)
            count += 1
        else:
            break
    cap.release()