import cv2
import numpy as np
import os
import single_calib as singlecalib
import utils

#kinect images
cam1_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\kinect\\imgs\\'
#canon images
cam2_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\canon\\'

#read calibration matrices
def get_matrices(cam_dir):
    k=np.load(os.path.join(cam_dir,'k.npy'))
    dist=np.load(os.path.join(cam_dir,'dist.npy'))
    return k,dist

cam1_k,cam1_dist=utils.get_calib_matrices(cam1_dir)
cam2_k,cam2_dist=utils.get_calib_matrices(cam2_dir)

cam1_files=[file for file in os.listdir(cam1_dir) if file.split('.')[-1].lower()=='jpg']
cam1_files.sort()
cam2_files=[file for file in os.listdir(cam2_dir) if file.split('.')[-1].lower()=='jpg']
cam2_files.sort()

objp=singlecalib.get_obj_points(d=40,grid_size=((4,11)))

objpoints = [] # 3d point in real world space
imgpoints_cam1 = [] # 2d points in image plane.
imgpoints_cam2 = []

for i in range(len(cam1_files)):
    cam1_points,sh1=singlecalib.get_points(os.path.join(cam1_dir,cam1_files[i]))
    cam2_points,sh2=singlecalib.get_points(os.path.join(cam2_dir,cam2_files[i]))
    if type(cam1_points)==np.ndarray and type(cam2_points)==np.ndarray:
        objpoints.append(objp)
        imgpoints_cam1.append(cam1_points)
        imgpoints_cam2.append(cam2_points)


#CALIB_FIX_INTRINSIC :  Fix cameraMatrix? and distCoeffs? so that only R, T, E , and F matrices are estimated.
#cv2.CALIB_USE_INTRINSIC_GUESS : Optimize some or all of the intrinsic parameters according to the specified flags. Initial values are provided by the user.

stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC and cv2.CALIB_USE_INTRINSIC_GUESS
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_cam1, imgpoints_cam2, 
                                                              cam1_k, cam1_dist,
                                                              cam2_k, cam2_dist, (1,1),
                                                              criteria = criteria, flags = stereocalibration_flags)

out_path='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\'
np.save(os.path.join(out_path,'R.npy'),R)
np.save(os.path.join(out_path,'T.npy'),T)










        












