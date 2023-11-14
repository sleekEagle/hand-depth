import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from KyptPred import PredictKypt
# import DetectTime
import json
import argparse
import cv2
import numpy as np 
from detect_kypts import utils
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from detect_kypts import utils
import ast

#read the calibration matrices
kinect_calib_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\kinect\\imgs\\'
canon_calib_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\canon\\'
k_kinect,dist_kinect=utils.get_calib_matrices(kinect_calib_dir)
k_canon,dist_canon=utils.get_calib_matrices(canon_calib_dir)

#get stereo transformation matrix
stereo_calib_path='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata4\\calib\\'
R,T=utils.get_stereo_calib_matrices(stereo_calib_path)


k_kinect=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\kinect_k.txt')
k_canon=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\canon_k.txt')
R=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\R.txt')
T=utils.get_matr(r'C:\Users\lahir\data\calibration\nov13\T.txt')

#test the  R and T matrices
sys.path.append('C:\\Users\\lahir\\code\\hand-depth\\detect_kypts\\')
import single_calib as singlecalib
import matplotlib.pyplot as plt

kinect_img=r'C:\Users\lahir\data\kinect_hand_data\test\color\65.jpg'
kinect_depth=r'C:\Users\lahir\data\kinect_hand_data\test\depth\65.png'
canon_img=r'C:\Users\lahir\Pictures\2023_11_13\65.jpg'
kinect = cv2.imread(kinect_img, cv2.IMREAD_COLOR)
depth = cv2.imread(kinect_depth, cv2.IMREAD_UNCHANGED).astype('float32')


kinect_pts=singlecalib.get_points(kinect_img)[0]
plt.imshow(kinect)
plt.scatter(kinect_pts[:,0,0],kinect_pts[:,0,1])
plt.show()

plt.imshow(depth)
plt.scatter(kinect_pts[:,0,0],kinect_pts[:,0,1])
plt.show()

x_ar,y_ar,d_ar=[],[],[]
canon_2d_ar=np.empty((0,2))
for i in range(kinect_pts.shape[0]):
    x,y=kinect_pts[i][0]
    d_=depth[int(y),int(x)]
    d_ar.append(d_)
    x_ar.append(x)
    y_ar.append(y)
    X,Y,Z=utils.calc_XYZ(k_kinect,x,y,d_)
    coord=np.array([[X,Y,Z]]).swapaxes(1,0)
    #trandform to cannon 
    proj=np.matmul(R,coord)+T.T
    cancon_homo=np.matmul(k_canon,proj)
    cancon_homo=cancon_homo/cancon_homo[-1]
    cancon_2d=cancon_homo[:-1,:].swapaxes(1,0)
    canon_2d_ar=np.concatenate((canon_2d_ar,cancon_2d),axis=0)

cannon_img=cv2.imread(canon_img, cv2.IMREAD_COLOR)
plt.imshow(cannon_img)
plt.scatter(canon_2d_ar[:,0],canon_2d_ar[:,1])
plt.show()

    

depth[kinect_pts[:,0,0].astype(int),:]


#read kinect timestamp and depth values of the keypoints
data_file=r'C:\Users\lahir\data\kinect_hand_data\testdata2\data.json'
f=open(data_file)
d=f.read()
data=json.loads(d)
data=ast.literal_eval(data)

k_depths=np.empty((0,21))
ts_kinect=[]
XYZ_ar=np.empty((0,21,3))
for key in data.keys():
    ts_=data[key]['ts']
    ts_kinect.append(ts_)
    k_depth=np.array([data[key]['keypoint_depths']])
    k_depths=np.concatenate((k_depths,k_depth),axis=0)

    kypt_coords=[]
    cannon_coords=np.empty((0,2))
    for n in range(21):
        #get 3D XYZ positions
        x,y=data[key]['keypoints'][n]
        d=data[key]['keypoint_depths_interp'][n] 
        X,Y,Z=utils.calc_XYZ(k_kinect,x,y,d)
        kypt_coords.append([X,Y,Z])
        #transform to canon coordinate space
        kinect_coord=np.array([[X*1e3,Y*1e3,Z*1e3]]).swapaxes(1,0)
        proj=np.matmul(R,kinect_coord)+T
        #project to canon image plane
        cancon_homo=np.matmul(k_canon,proj)
        cancon_homo=cancon_homo/cancon_homo[-1]
        cancon_2d=cancon_homo[:-1,:].swapaxes(1,0)
        cannon_coords=np.concatenate((cannon_coords,cancon_2d),axis=0)

    XYZ_ar=np.concatenate((XYZ_ar,np.array([kypt_coords])))

pt_x,pt_y=cannon_coords[:,0],cannon_coords[:,1]
imgpth=r'C:\Users\lahir\data\kinect_hand_data\testdata2\cannon\20000101000111_IMG_0346.jpg'
img=cv2.imread(imgpth,cv2.IMREAD_COLOR)
import matplotlib.pyplot as plt
plt.imshow(img)
# plt.scatter(pt_x,pt_y)
plt.scatter(cannon_coords[:,0],cannon_coords[:,1])
plt.show()


#get canon timstamps
data_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata2\\cannon\\'
files=utils.get_files_ext(data_dir,['jpg'])
ts_ms_list=[]
for file in files:
    ts=utils.get_ts_google(os.path.join(data_dir,file))
    ts_ms=utils.get_ms(ts)
    ts_ms_list.append(ts_ms)
ts_s=np.array(ts_ms_list)/1000

#interpolate kinect depths and keypoint loations at the cannot timestamps
interp_funcs_x=utils.fit_cubicsplines(np.array(ts_kinect),XYZ_ar[:,:,0])
interp_funcs_y=utils.fit_cubicsplines(np.array(ts_kinect),XYZ_ar[:,:,1])
interp_funcs_z=utils.fit_cubicsplines(np.array(ts_kinect),XYZ_ar[:,:,2])

interp_XYZ_ar=np.empty((ts_s.shape[0],0,3))
for i in range(XYZ_ar.shape[1]):
    X_interp_coords=interp_funcs_x[i](ts_s)
    Y_interp_coords=interp_funcs_y[i](ts_s)
    Z_interp_coords=interp_funcs_z[i](ts_s)
    XYZ_interp=np.array([[X_interp_coords,Y_interp_coords,Z_interp_coords]])
    XYZ_interp=np.moveaxis(XYZ_interp,[0,1,2],[1,2,0])
    interp_XYZ_ar=np.concatenate((interp_XYZ_ar,XYZ_interp),axis=1)

#test: project points back to image plane
pt3d=np.array([XYZ_ar[3,4,:]]).swapaxes(1,0)
proj=np.matmul(k_kinect,pt3d)
proj=proj/[proj[-1]]
data[3]['keypoints'][4]

#project to cannot coord




XYZ_ar_=XYZ_ar.reshape(-1,3)
proj=np.matmul(k_kinect,XYZ_ar_.T)
proj_x=proj[0,:]/proj[-1,:]
proj_y=proj[1,:]/proj[-1,:]

proj_x=proj_x.reshape(-1,85)
proj_y=proj_y.reshape(-1,85)
proj_xy=np.array([proj_x,proj_y])

kypts=np.array(data[0]['keypoints']).swapaxes(1,0)
proj_xy[:,:,0]-kypts






interp_XYZ_ar=interp_XYZ_ar.reshape((-1,3)) 
T_rep=np.repeat(T,repeats=interp_XYZ_ar.shape[0],axis=1)
proj_pts=np.matmul(R,interp_XYZ_ar.T) + T_rep
depths=np.sqrt(np.square(proj_pts[0,:]) + np.square(proj_pts[1,:]) + np.square(proj_pts[2,:]))

#project 3D points to the camera plane
img_pts=np.matmul(k_canon,proj_pts)
img_pts=np.swapaxes(img_pts,0,1)
img_pts=img_pts.reshape(-1,21,3)
#unhomogenize the points
img_pts_x=img_pts[:,:,0]/img_pts[:,:,-1]
img_pts_y=img_pts[:,:,1]/img_pts[:,:,-1]

img=cv2.imread(os.path.join(data_dir,files[0]),cv2.IMREAD_COLOR)
import matplotlib.pyplot as plt
plt.imshow(img)
plt.scatter(img_pts_x[0,:],img_pts_y[0,:])
plt.show()


utils.show_img(img)




























n=XYZ_ar.shape[0]
T_rep=np.repeat(T,repeats=n,axis=1)
proj_pts=np.matmul(R,XYZ_ar.T) + T_rep

#get d
d_ar=np.sqrt(np.square(XYZ_ar[:,0])+np.square(XYZ_ar[:,1])+np.square(XYZ_ar[:,2]))
d_ar=d_ar.reshape(-1,21)

#project to canon image
k_canon.shape



ts_kinect=np.array(ts_kinect)
func_list=utils.fit_cubicsplines(ts_kinect,k_depths)

pred=func_list[20](ts_kinect)

plt.plot(ts_kinect,k_depths[:,20])
plt.plot(ts_kinect,pred)
plt.show()


data_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata2\\cannon\\'
files=utils.get_files_ext(data_dir,['jpg'])
ts_ms_list=[]
for file in files:
    ts=utils.get_ts_google(os.path.join(data_dir,file))
    ts_ms=utils.get_ms(ts)
    ts_ms_list.append(ts_ms)
ts_s=np.array(ts_ms_list)/1000

pred=func_list[20](ts_s)
plt.plot(ts_s,pred)
plt.show()

pred_kypt_depths=np.empty((0,k_depths.shape[1]))

    #interpolate the depth of each keypoint in kinnect coordinate system
    kp_depths=[]
    for i in range(k_depths.shape[1]): 
        kp_depths.append(func_list[i](ts_ms).item())
    kp_depths=np.array([kp_depths])
    pred_kypt_depths=np.concatenate((pred_kypt_depths,kp_depths),axis=0)

plt.plot(ts_ms_list,pred_kypt_depths[:,4])
plt.show()

    
for file in files:
    ts=utils.get_ts_google(os.path.join(data_dir,file))
    ts_list.append(ts)



func_list=utils.fit_cubicsplines(ts,k_depths)
utils.get_ms(ts_list[0])
func_list[0](utils.get_ms(ts_list[0]))



#interpolate missing values of the k_depths
filled_depths=np.empty((k_depths.shape[0],0))
for i in range(k_depths.shape[-1]):
    vals=k_depths[:,i]
    args=np.argwhere(vals==0)[:,0]
    pred_vals=func_list[i](ts)
    filled_vals=vals.copy()
    filled_vals[args]=pred_vals[args]
    filled_depths=np.concatenate((filled_depths,np.expand_dims(filled_vals,axis=1)),axis=-1)



func_list[3](ts)



spl = CubicSpline([1, 2, 3, 4, 5, 6], [1, 4, 8, 16, 25, 36])
spl(3.5)


plt.plot(ts)
plt.show()

plt.plot(ts,k_depths[:,5])
plt.show()



f = interp1d(ts, k_depths, kind='cubic',axis=0)
f(60670460.0)

plt.plot(ts,k_depths[:,1])
plt.show()







from datetime import datetime

ts_ms=get_ms(date_time_obj)


# Define the sine function
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)

f = interp1d(x, y, kind='cubic')
x_new = np.linspace(0, 2*np.pi, 100)

y_new = f(x_new)

plt.plot(x_new,y_new)
plt.show()

plt.plot(x,y)
plt.show()







