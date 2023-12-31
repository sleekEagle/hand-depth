import cv2
import numpy as np
import os

photos_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\calibration\\canon\\'
out_path='C:\\Users\\lahir\\data\\kinect_hand_data\\calibration\\canon\\'

def get_reprojection_errors(objpoints,imgpoints,rvecs,tvecs,mtx,dist):
    errors=[]
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        errors.append(error)
    errors=np.array(errors)
    print( "mean error: {}".format(np.mean(errors)))
    return errors

'''
calibrate with an asymetric circular grid.
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


def get_points(img_pth,d=40,grid_size=(4,11)):
    gray=cv2.imread(img_pth,cv2.IMREAD_GRAYSCALE)
    #detect circle centers
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findCirclesGrid(gray, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        im_with_keypoints = cv2.drawChessboardCorners(gray, (4,11), corners2, ret)
        im_ = cv2.resize(im_with_keypoints, (960, 540))    
        cv2.imshow("img", im_) # display
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return corners2,gray.shape
    else:
        return -1,gray.shape
    
def calibrate(photos_dir,d=42.5,grid_size=(4,11)):
    #calibrating with the asymetric circular grid 
    objp=get_obj_points(d,grid_size)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    found=0

# p=r'C:\Users\***\data\pixelcalib\telephoto\OpenCamera\IMG_20230829_065058_1.jpg'
# gray=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    img_pths=[file for file in os.listdir(photos_dir) if file.split('.')[-1].lower()=='jpg']
    for i,item in enumerate(img_pths):
        corners2,shp=get_points(os.path.join(photos_dir,item),d=d,grid_size=grid_size)
        if type(corners2)==np.ndarray:
            objpoints.append(objp)
            imgpoints.append(corners2)
            found += 1
            print(str(i)+' done')
        else:
            print(f'no corners detected in this image : {item}')

    print('num good images:'+str(found))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shp[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs,objpoints,imgpoints,shp[::-1]

#calibrate the first round
if __name__ == "__main__":
    ret, mtx, dist, rvecs,tvecs,objpoints,imgpoints,imshape=calibrate(photos_dir,d=40)
    errors=get_reprojection_errors(objpoints,imgpoints,rvecs,tvecs,mtx,dist)

    thr_error=0.008
    valid_args=np.argwhere(errors<thr_error)
    #recalibrate with the selected images
    objpoints_selected=[pt for i,pt in enumerate(objpoints) if i in valid_args]
    imgpts_selected=[pt for i,pt in enumerate(imgpoints) if i in valid_args]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_selected, imgpts_selected,imshape, None, None)
    errors=get_reprojection_errors(objpoints_selected,imgpts_selected,rvecs,tvecs,mtx,dist)

    #save the calibration matrices
    np.save(os.path.join(out_path,'k.npy'),mtx)
    np.save(os.path.join(out_path,'dist.npy'),dist)