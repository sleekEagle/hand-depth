import os
import sys
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
# from KyptPred import PredictKypt
# import DetectTime
import json
import argparse
import cv2
import numpy as np 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
os.path.dirname(sys.path[0])
sys.path.append(os.path.dirname(sys.path[0]))
print(sys.path)
import utils
import ast


#extract images from video
canon_dir='C:\\Users\\lahir\\data\\CPR_experiment\\test\\canon\\'
image_file=r'C:\\Users\\lahir\\data\\CPR_experiment\\test\\canon\\img_0505.MOV'
utils.extract_imgs(image_file,fps=60)

# #read the calibration matrices
# kinect_calib_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\kinect\\imgs\\'
# canon_calib_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\canon\\'
# k_kinect,dist_kinect=utils.get_calib_matrices(kinect_calib_dir)
# k_canon,dist_canon=utils.get_calib_matrices(canon_calib_dir)

# #get stereo transformation matrix
# stereo_calib_path='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\'
# R,T=utils.get_stereo_calib_matrices(stereo_calib_path)


k_kinect=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\kinect_k.txt')
k_canon=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\canon_k.txt')
R=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\R.txt')
T=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\T.txt')

#test the  R and T matrices
sys.path.append('C:\\Users\\lahir\\code\\hand-depth\\detect_kypts\\')
import single_calib as singlecalib
import matplotlib.pyplot as plt

# kinect_img=r'C:\Users\lahir\data\CPR_experiment\test\kinect\color\163.jpg'
# kinect_depth=r'C:\Users\lahir\data\CPR_experiment\test\canon\'
# canon_img=r'C:\Users\lahir\Pictures\2023_11_13\65.jpg'
# kinect = cv2.imread(kinect_img, cv2.IMREAD_COLOR)
# depth = cv2.imread(kinect_depth, cv2.IMREAD_UNCHANGED).astype('float32')


# kinect_pts=singlecalib.get_points(kinect_img)[0]
# plt.imshow(kinect)
# plt.scatter(kinect_pts[:,0,0],kinect_pts[:,0,1])
# plt.show()

# plt.imshow(depth)
# plt.scatter(kinect_pts[:,0,0],kinect_pts[:,0,1])
# plt.show()

# x_ar,y_ar,d_ar=[],[],[]
# canon_2d_ar=np.empty((0,2))
# for i in range(kinect_pts.shape[0]):
#     x,y=kinect_pts[i][0]
#     d_=depth[int(y),int(x)]
#     d_ar.append(d_)
#     x_ar.append(x)
#     y_ar.append(y)
#     X,Y,Z=utils.calc_XYZ(k_kinect,x,y,d_)
#     coord=np.array([[X,Y,Z]]).swapaxes(1,0)
#     #trandform to cannon 
#     proj=np.matmul(R,coord)+T.T
#     cancon_homo=np.matmul(k_canon,proj)
#     cancon_homo=cancon_homo/cancon_homo[-1]
#     cancon_2d=cancon_homo[:-1,:].swapaxes(1,0)
#     canon_2d_ar=np.concatenate((canon_2d_ar,cancon_2d),axis=0)

# cannon_img=cv2.imread(canon_img, cv2.IMREAD_COLOR)
# plt.imshow(cannon_img)
# plt.scatter(canon_2d_ar[:,0],canon_2d_ar[:,1])
# plt.show()

    

# depth[kinect_pts[:,0,0].astype(int),:]


#read kinect timestamp and depth values of the keypoints
data_file=r'C:\Users\lahir\data\CPR_experiment\test\kinect\sorted_data.json'
f=open(data_file)
d=f.read()
data=json.loads(d)
data=ast.literal_eval(data)

k_depths=np.empty((0,21))
ts_kinect=[]
img_num=[]

XYZ_ar,canon_XYZ_ar=np.empty((0,21,3)),np.empty((0,21,3))
canon_depths_ar=np.empty((0,21))
for key in data.keys():
    ts_=data[key]['ts']
    img_num_=data[key]['img_num'] 
    img_num.append(img_num_)
    ts_kinect.append(ts_)
    k_depth=np.array([data[key]['keypoint_depths']])
    k_depths=np.concatenate((k_depths,k_depth),axis=0)

    kypt_coords=[]
    cannon_coords=np.empty((0,3))
    
    for n in range(21):
        #get 3D XYZ positions
        x,y=data[key]['keypoints'][n]
        d=data[key]['keypoint_depths_interp'][n] 
        X,Y,Z=utils.calc_XYZ(k_kinect,x,y,d)
        kypt_coords.append([X,Y,Z])
        #transform to canon coordinate space
        kinect_coord=np.array([[X*1e3,Y*1e3,Z*1e3]]).swapaxes(1,0)
        proj=np.matmul(R,kinect_coord)+T
        cannon_coords=np.concatenate((cannon_coords,proj.T),axis=0)
        # #project to canon image plane
        # cancon_homo=np.matmul(k_canon,proj)
        # cancon_homo=cancon_homo/cancon_homo[-1]
        # cancon_2d=cancon_homo[:-1,:].swapaxes(1,0)
        # cannon_coords=np.concatenate((cannon_coords,cancon_2d),axis=0)

    
    depths=np.sqrt(np.square(cannon_coords[:,0])+np.square(cannon_coords[:,1])+np.square(cannon_coords[:,2]))
    depths=np.array([depths])
    canon_depths_ar=np.concatenate((canon_depths_ar,depths),axis=0)
    XYZ_ar=np.concatenate((XYZ_ar,np.array([kypt_coords])))
    canon_XYZ_ar=np.concatenate((canon_XYZ_ar,np.array([cannon_coords])))


#sort based on ts
sorted_indices = np.argsort(np.array(img_num))
sorted_img_num=np.array(img_num)[sorted_indices]
ts_kinect=np.array(ts_kinect)[sorted_indices]
canon_depths_ar=canon_depths_ar[sorted_indices,:]

np.argwhere(sorted_img_num==56)


#testing******************************************
i=16
cancon_homo=np.matmul(k_canon,canon_XYZ_ar[i].T)
cancon_homo=cancon_homo/cancon_homo[-1]
canon_img=r'C:\Users\lahir\data\CPR_experiment\test\canon\imgs\image-1576.jpg'
img=cv2.imread(canon_img,cv2.IMREAD_COLOR)
for i in range(21):
    cv2.circle(img, (round(cancon_homo[0,i]),round(cancon_homo[1,i])), radius=5, color=(0, 0, 255), thickness=-1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#end testing*******************************

# imgpth=r'C:\Users\lahir\data\CPR_experiment\test\canon\imgs\image-1705.jpg'
# utils.get_ts_google(imgpth)


#get canon timstamps
# canon_img_dir=os.path.join(canon_dir,'imgs')
# files=utils.get_files_ext(canon_img_dir,['jpg'])
# ts_ms_list=[]
# ts_detected_files=[]
# for i,file in enumerate(files):
#     ts=utils.get_ts_google(os.path.join(canon_img_dir,file))
#     if type(ts) is str:
#         try:
#             ts_ms=utils.get_ms(ts)
#             ts_ms_list.append(ts_ms)
#             ts_detected_files.append(file)
#             print(f'ts: {ts} done:{i/len(files)}')
#         except:
#             print('Could not extract ts from img text due to ecception')
#     else:
#         print(f'ts not detected in image {file}')
# ts_s=np.array(ts_ms_list)/1000

# #ave these (for degubbing)
# import pickle
# with open('C:\\Users\\lahir\\data\\CPR_experiment\\test\\canon\\ts_s.pkl', 'wb') as file:
#     pickle.dump(ts_s, file)
# with open('C:\\Users\\lahir\\data\\CPR_experiment\\test\\canon\\files.pkl', 'wb') as file:
#     pickle.dump(ts_detected_files, file)

print('here')
import pickle
with open('C:\\Users\\lahir\\data\\CPR_experiment\\test\\canon\\ts_s.pkl', 'rb') as file:
    ts_s = pickle.load(file)
with open('C:\\Users\\lahir\\data\\CPR_experiment\\test\\canon\\files.pkl', 'rb') as file:
    ts_detected_files = pickle.load(file)


#start and end positions of the section we want to interpolate
k_start,k_end=14,114
ts_kinect_int=ts_kinect[k_start:k_end]
canon_depths_ar_inter=canon_depths_ar[k_start:k_end]
#interpolate kinect depths and keypoint loations at the cannot timestamps
interp_funcs=utils.fit_cubicsplines(ts_kinect_int,canon_depths_ar[k_start:k_end,:])

min_k_ts,max_k_ts=np.min(ts_kinect_int),np.max(ts_kinect_int)
ts_s_int=ts_s[ts_s>min_k_ts]
ts_s_int=ts_s_int[ts_s_int<max_k_ts]


start,end=0,100
t=ts_kinect_int[start:end]
interp_funcs=utils.fit_cubicsplines(t,canon_depths_ar_inter[start:end,:])
t_int=np.linspace(np.min(t),np.max(t),num=500)
int_vals=interp_funcs[0](t_int)
plt.plot(ts_kinect_int[start:end],canon_depths_ar_inter[start:end,0])
plt.plot(t_int,int_vals)
plt.show()









interp_dept_ar=np.empty((0,ts_s_int.shape[0]))
for i in range(XYZ_ar.shape[1]):
    X_interp_coords=interp_funcs[i](ts_s_int)
    X_interp_coords=np.array([X_interp_coords])
    interp_dept_ar=np.concatenate((interp_dept_ar,X_interp_coords),axis=0)

intcoords=interp_funcs[0](ts_kinect_int)
plt.scatter(ts_kinect_int,canon_depths_ar[k_start:k_end,0])
plt.scatter(ts_kinect_int,intcoords)
plt.show()

plt.scatter(ts_s_int,interp_dept_ar[0,:])
plt.plot(ts_kinect_int,canon_depths_ar[k_start:k_end,0],'-o',c='r')
plt.show()