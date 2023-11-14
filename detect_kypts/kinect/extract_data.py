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
import utils
from pathlib import Path

print('done imports')

def get_ts(ts_str):
    splt=ts_str.split('_')
    ts=float(splt[0])*60*60 + float(splt[1])*60 + float(splt[2])*1
    return ts

# fps: camera FPS used
def main(args,fps=5):
    rgb_dir=os.path.join(args.data_dir,'color')
    depth_dir=os.path.join(args.data_dir,'depth')

    if args.save_kypt_imgs:
        kypt_pth=os.path.join(args.data_dir,'kypt_plots')
        Path(kypt_pth).mkdir(parents=True, exist_ok=True)

    #extract start timestamp
    rec_name=[file for file in os.listdir(args.data_dir) if file[-3:]=='mkv']
    tsfile_name=os.path.join(args.data_dir,rec_name[0].replace('mkv','txt'))

    with open(tsfile_name, 'r') as file:
        lines = file.readlines()
    ts_list=[line.replace('\n','') for line in lines]
    ts_s_list=np.array([get_ts(ts_str) for ts_str in ts_list])

    #if bbox_score > bb_thresh no hands are detected
    bb_thresh=0.8
    Kypt=PredictKypt()
    data={}
    rgb_files=[file for file in os.listdir(rgb_dir) if file.split('.')[-1]=='jpg']
    n=0
    k_depths_ar=np.empty((0,21))
    for i,file in enumerate(rgb_files):
        full_path=os.path.join(rgb_dir,file)
        result=Kypt.get_kypts(full_path)
        bb_score=result['bbox_score']
        img_num=int(file.split('.')[0])
        ts_num=img_num-1
        if args.save_kypt_imgs:
            Kypt.save_kypts(os.path.join(kypt_pth,file))

        #read depth image
        d_file=file.split('.')[0]+'.png'
        depth = cv2.imread(os.path.join(depth_dir,d_file), cv2.IMREAD_UNCHANGED).astype('float32')
        depth = depth / 1000.0  # convert to meters

        if bb_score<bb_thresh:
            #hand detected
            hand_kypts=result['keypoints']
            #get depth for each keypoint location
            k_depths=[]
            #save an image with keypoints shown


            for kypt in hand_kypts:
                kx,ky=kypt
                #get depth average of 3 pixel radius around the keypoint location
                r=10
                d_sel=depth[int(ky-r):int(ky+r),int(kx-r):int(kx+r)]
                #remove unvalid depth values
                d_sel=d_sel[d_sel>0]
                if d_sel.shape[0]==0:
                    k_depth=0
                else:
                    k_depth=np.mean(d_sel)
                k_depths.append(k_depth)


            bb=result['bbox']
            this_data={'img_num': img_num,
                    'keypoints': result['keypoints'],
                    'bbox': result['bbox'],
                    'keypoint_scores': result['keypoint_scores'],
                    'bbox_score': result['bbox_score'],
                    'keypoint_depths': k_depths,
                    'ts': ts_s_list[ts_num]}
            k_depths_ar=np.concatenate((k_depths_ar,np.array([k_depths])),axis=0)
            # if detect_time:
            #     #extract timestamp from digital clock with computer vision
            #     ts=DetectTime.get_ts_from_image(full_path,clock_coord)
            #     this_data['clock_ts']=ts

            data[n]=this_data
            n+=1

    #interpolate missing depth values : cubic spline interpolation

    interp_funcs=utils.fit_cubicsplines(ts_s_list,k_depths_ar)
    #interpolate missing values of the k_depths
    filled_depths=np.empty((k_depths_ar.shape[0],0))
    for i in range(k_depths_ar.shape[-1]):
        vals=k_depths_ar[:,i]
        indx=np.argwhere(vals==0)[:,0]
        pred_vals=interp_funcs[i](ts_s_list)
        filled_vals=vals.copy()
        filled_vals[indx]=pred_vals[indx]
        filled_depths=np.concatenate((filled_depths,np.expand_dims(filled_vals,axis=1)),axis=-1)
    
    #add the interpolated values to the data dictionary
    for i,key in enumerate(data.keys()):
        data[key]['keypoint_depths_interp']=list(filled_depths[i,:])

    json_file=os.path.join(args.data_dir,'data.json')
    with open(json_file, "w") as outfile: 
        json.dump(str(data), outfile) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect hand keypoints')
    parser.add_argument('--data_dir', type=str, help='directory containing data',
                        default='C:\\Users\\lahir\\data\\CPR_experiment\\test\\kinect\\')
    parser.add_argument('--save_kypt_imgs', type=bool, help='save images with keypts shown?',
                        default=True)
    args = parser.parse_args()
    main(args)  

# import json
# import numpy as np
# import ast
# import matplotlib.pyplot as plt

# f=open(r'C:\Users\lahir\data\CPR_experiment\test\kinect\data.json')
# d=f.read()
# data=json.loads(d)
# data=ast.literal_eval(data)

# d_list=[]
# d_list_interp=[]
# for i in range(len(data)):
#     d=data[i]['keypoint_depths']
#     d_interp=data[i]['keypoint_depths_interp']
#     d_list.extend(d)
#     d_list_interp.extend(d_interp)

# plt.plot(d_list)
# plt.plot(d_list_interp)
# plt.show()


# import json
# outfile=r'C:\Users\lahir\data\kinect_hand_data\CPR_data\kinect\session1\data.json'
# d={}
# d[0]={'a':[1,2,3],'b':'nothing'}
# d[1]={'g':'ff','h':[[1,2],[5,4]]}
# d[2]={'gf':([1,2,4],[4,5])}
# with open(outfile, 'w') as f:
#     json.dump((d), f, indent=4)

# f=open(outfile)
# data=json.load(f)


# import time
# time_str = "2023-10-31 12:30:00.123"
# # Convert the time string to a struct_time object
# time_struct = time.strptime(time_str, "%Y-%m-%d %H:%M:%S.ms")

# # Convert the struct_time object to epoch time
# epoch_time = (time.mktime(time_struct))