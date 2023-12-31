import cv2
# import pytesseract
import re
from scipy.interpolate import CubicSpline
import numpy as np
from datetime import datetime
import os


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def show_img(image):
    image=cv2.resize(image,(500,500),interpolation = cv2.INTER_AREA)
    cv2.imshow('Detected Rectangles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
use pytesseract 
img: opencv greyscale image 
clock_coord: pixle coordinates of the clock location
use this as follows: 
give path to image:
    impth='C:\\Users\\lahir\\data\\kinect_hand_data\\CPR_data\\kinect\\frames\\color\\00037.jpg'
    clock_coord=[226,192,412,246]
    get_ts_from_image(impth,clock_coord)
give opencv image read as grayscale:
    img = cv2.imread(impth,cv2.IMREAD_GRAYSCALE)
    clock_coord=[226,192,412,246]
    get_ts_from_image(impth,clock_coord)
'''
def get_ts_from_image(img,clock_coord):
    if type(img)==str:
        img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    clock_img=img[clock_coord[1]:clock_coord[3],clock_coord[0]:clock_coord[2]]
    thresh = cv2.threshold(clock_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')
    numbers=re.findall("\d+", data) 
    assert len(numbers)==3, "time detection failed"
    ts='.'.join(numbers)
    return ts

# clock_coord=[1107,132,1821,209]
# get_ts_from_image(img,clock_coord)

def extract_time_str(txt):
    x = re.findall("[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]", txt)
    if len(x)>0:
        return x[0]
    else: 
         return None

#extract text from image using Google Viaion API
'''
to use:
img=r'C:\\Users\\lahir\\Downloads\\time.jpg'
get_ts_google(img)
'''
def get_ts_google(image_path):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
            content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        match=extract_time_str(text.description)
        if type(match)==str:
            break
    return match

# get_ts_google(r'C:\Users\lahir\Downloads\Photos-001\20000101000129_IMG_0256.JPG')

def fit_cubicsplines_nozeros(x,val_ar):
    func_list=[]
    for i in range(val_ar.shape[-1]):
        vals=val_ar[:,i]
        args=np.argwhere(vals>0)
        x_=x[args][:,0]
        vals_=vals[args][:,0]
        spl=CubicSpline(x_,vals_)
        func_list.append(spl)
    return func_list


def fit_cubicsplines(x,val_ar):
    func_list=[]
    for i in range(val_ar.shape[-1]):
        vals=val_ar[:,i]
        spl=CubicSpline(x,vals)
        func_list.append(spl)
    return func_list

#convert datetime timestamp to milliseconds
def get_ms(ts):
    t = datetime.strptime(ts, "%H:%M:%S.%f")
    return t.hour*1000*60*60 + t.minute*1000*60 + t.second*1000 + t.microsecond/1000

#extract files with one of the given set of extentions 
def get_files_ext(dir_pth,possible_ext=['jpg','png']):
    files=[file for file in os.listdir(dir_pth) if file.split('.')[-1].lower() in possible_ext]
    return files

def show_cvimg(img):
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() # It destroys the image showing the window.

#read calibration matrices
def get_calib_matrices_npy(cam_dir):
    k=np.load(os.path.join(cam_dir,'k.npy'))
    dist=np.load(os.path.join(cam_dir,'dist.npy'))
    return k,dist

#get stereo calibration matrices
def get_stereo_calib_matrices_npy(stereo_calib_path):
    R=np.load(os.path.join(stereo_calib_path,'R.npy'))
    T=np.load(os.path.join(stereo_calib_path,'T.npy'))
    return R,T

'''
get matrix from text file
a,b,c
d,f,g
v,b,n
'''
def get_matr(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    matr=[]
    for line in lines:
        line_spl=line.split(',')
        vals=[]
        for l in line_spl:
            vals.append(float(l.replace('\n','')))
        matr.append(vals)
    return np.array(matr)


'''
get the 3D coordinates (XYZ) given 
camera projection matrix, pixel (x,y) coordinates and 
depth (XYZ units are in the same units as d)
'''
def calc_XYZ(k,x,y,d):
    cx,cy=k[0,2],k[1,2]
    fx,fy=k[0,0],k[1,1]
    Z=d/(((x-cx)/fx)**2+((y-cy)/fy)**2+1)**0.5
    X=(x-cx)/fx*Z
    Y=(y-cy)/fy*Z
    return X,Y,Z

#extract on image from a video, do this for all the videos
def extract_imgs_mul(data_dir,ext='mov'):
    import subprocess
    outdir='imgs'
    os.makedirs(os.path.join(data_dir,outdir), exist_ok=True)
    mov_files=[file for file in os.listdir(data_dir) if ext.lower() in file.lower()]
    mov_files.sort()
    for i,file in enumerate(mov_files):
        out_file=f'{i+1}.jpg'
        cd=f'ffmpeg -i {os.path.join(data_dir,file)} -vframes 1 {os.path.join(data_dir,outdir,out_file)}'
        result = subprocess.run(cd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#extract frames from a given video
# video_file=r'C:\Users\lahir\data\CPR_experiment\test\canon\img_0505.MOV'
def extract_imgs(video_file,fps=60):
    import subprocess
    outdir='imgs'
    dir=os.path.dirname(video_file)
    outdir=os.path.join(dir,'imgs')
    os.makedirs(os.path.join(dir,outdir), exist_ok=True)
    cd=f'ffmpeg -i {video_file} -r {fps} -q:v 2 {os.path.join(dir,outdir)}\\image-%03d.jpg'
    result = subprocess.run(cd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def extract_kinect_imgs(args):
    import open3d as o3d
    from PIL import Image
    v=args['pad']
    print(f'args.pad : {v}')

    #convert parseargs to dict

    SIZE=(1920,1280)
    
    reader = o3d.io.AzureKinectMKVReader()
    reader.open(args['record_filename'])

    color_dir=os.path.join(args['output'],'color')
    depth_dir=os.path.join(args['output'],'depth')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(color_dir)
    print(depth_dir)
    try:
        os.makedirs(color_dir, exist_ok=False)
        os.makedirs(depth_dir, exist_ok=False)
    except Exception as e:
        print("Error:", e)
    
    #read existing color images
    img_n=len([file for file in os.listdir(color_dir) if file.split('.')[-1].lower()=='jpg'])
    print(f'existing images = {img_n}')

    while not reader.is_eof():
        rgbd = reader.next_frame()
        if rgbd is None:
            continue
        if args['output'] is not None:
            # print('here')
            img_n+=1
            np_img=np.array(rgbd.color)
            # print(f'image size {np_img.shape}')
            col_image=Image.fromarray(np_img,'RGB')
            #pad image
            print(args['pad'])
            if args['pad']:
                result = Image.new(col_image.mode, SIZE, (0, 0, 0)) 
                result.paste(col_image, (0,0)) 
            else:
                result=col_image
            result.save(os.path.join(color_dir,str(img_n)+'.jpg'))
            depthimg=Image.fromarray(np.asarray(rgbd.depth))
            depthimg.save(os.path.join(depth_dir,str(img_n)+'.png'))
            print(f'{img_n} image saved')


























