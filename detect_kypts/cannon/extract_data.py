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
import datetime
from detect_kypts import DetectTime

data_dir='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata1\\cannon\\color\\'
files=[file for file in os.listdir(data_dir) if file.split('.')[-1].lower()=='jpg']


for file in files:
    ts=DetectTime.get_ts_google(os.path.join(data_dir,file))
    


