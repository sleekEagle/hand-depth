import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def show_img(image):
    image=cv2.resize(image,(500,500),interpolation = cv2.INTER_AREA)
    cv2.imshow('Detected Rectangles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
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







