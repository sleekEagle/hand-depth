import cv2
import numpy as np
import os

calib_mtx_pth='D:\\data\\calibration\\kinect_calib\\kinect\\k.npy'
dist_mtx_pth='D:\\data\\calibration\\kinect_calib\\kinect\\dist.npy'
img_dir='D:\\data\\calibration\\kinect_calib\\kinect\\rgb\\'

#grid parameters
grid_size=(4,11)
d=14

'''
calibrate with an asymetric circular grid.
get_obj_points : get 2D world coordinates of points on the circular pattern
z coordinates will be all-zero
d : distance between two adjecent circles (the longer distance) in mm
'''
def get_obj_points(d=42.5,grid_size=(4,11)):
    objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    #distance between two circle centers in mm 
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            indx=grid_size[0]*i+j
            x_value=i*d/2
            y_value=j*d + d/2*(i%2)
            objp[indx,:]=[x_value,y_value,0]
    return objp

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_undist_image(img_pth):
    img=cv2.imread(img_pth)
    #undistort
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return undst

def detect_centers(img):
    #convert image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findCirclesGrid(gray_image, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret:
        corners2 = cv2.cornerSubPix(gray_image,corners,(11,11),(-1,-1),criteria)
    return corners2

#read calibration and distortion matrices
mtx=np.load(calib_mtx_pth)
dist=np.load(dist_mtx_pth)  


#get all image files in the image dir
files=[os.path.join(img_dir,f) for f in os.listdir(img_dir) if ((f[-3:]=='png') or (f[-3:]=='jpg'))]
print('number of images = '+str(len(files)))

file=files[0]
undst=get_undist_image(file)
centers=detect_centers(undst)

#solve pnp to get R and T of the pattern wrt the camera
objp=get_obj_points(d=151)
ret,rvecs, tvecs = cv2.solvePnP(objp, centers, mtx, dist)

#detect hand keypoints
import kypt_transformer.main.predict as predict
from PIL import Image 

pre=predict.Predict()

#crop to be square and resize image to be 256x256
h,w=undst.shape
image=undst[0:h,0:h]
imge = Image.fromarray(image) 
image=imge.resize((256,256))
pre.make_prediction([image])

im=Image.open('D:\\data\\calibration\\kinect_calib\\kinect\\rgb\\1.jpg')
image=im.crop((0,0,256,256))
pre.make_prediction([image])

























