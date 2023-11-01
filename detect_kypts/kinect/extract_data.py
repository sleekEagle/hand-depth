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

# fps: camera FPS used
def main(data_dir,fps=5,detect_time=False,clock_coord=[]):
    rgb_dir=os.path.join(data_dir,'kinect','color')
    depth_dir=os.path.join(data_dir,'kinect','depth')

    #extract start timestamp
    rec_name=[file for file in os.listdir(data_dir) if file[-3:]=='mkv']
    if len(rec_name)>0:
        timestr=rec_name[0][7:-4]
        format_str = "%H_%M_%S.%f"
        start_time = datetime.datetime.strptime(timestr, format_str)

    #if bbox_score > bb_thresh no hands are detected
    bb_thresh=0.6
    Kypt=PredictKypt()
    data={}
    rgb_files=[file for file in os.listdir(rgb_dir) if file.split('.')[-1]=='jpg']
    n=0
    for file in rgb_files:
        full_path=os.path.join(rgb_dir,file)
        result=Kypt.get_kypts(full_path)
        bb_score=result['bbox_score']
        img_num=int(file.split('.')[0])
        delta=datetime.timedelta(seconds=1/fps*img_num)
        img_ts=(start_time+delta).time().strftime('%H:%M:%S.%f')

        #read depth image
        d_file=file.split('.')[0]+'.png'
        depth = cv2.imread(os.path.join(depth_dir,d_file), cv2.IMREAD_UNCHANGED).astype('float32')
        depth = depth / 1000.0  # convert to meters

        if bb_score<bb_thresh:
            #hand detected
            hand_kypts=result['keypoints']
            #get depth for each keypoint location
            k_depths=[]
            for kypt in hand_kypts:
                kx,ky=kypt
                #get depth average of 3 pixel radius around the keypoint location
                r=3
                d_sel=depth[int(kx-r):int(kx+r),int(ky-r):int(ky+r)]
                #remove unvalid depth values
                d_del=d_sel[d_sel>0]
                if d_del.shape[0]==0:
                    k_depth=0
                else:
                    k_depth=np.mean(d_del)
                k_depths.append(k_depth)


            bb=result['bbox']
            this_data={'img_num': img_num,
                    'keypoints': result['keypoints'],
                    'bbox': result['bbox'],
                    'keypoint_scores': result['keypoint_scores'],
                    'bbox_score': result['bbox_score'],
                    'keypoint_depths': k_depths,
                    'ts': img_ts}
            # if detect_time:
            #     #extract timestamp from digital clock with computer vision
            #     ts=DetectTime.get_ts_from_image(full_path,clock_coord)
            #     this_data['clock_ts']=ts

            data[n]=this_data
            n+=1

    json_file=os.path.join(data_dir,'data.json')
    with open(json_file, "w") as outfile: 
        json.dump(str(data), outfile) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect hand keypoints')
    parser.add_argument('--data_dir', type=str, help='directory containing data',
                        default='C:\\Users\\lahir\\data\\kinect_hand_data\\testdata1\\')
    parser.add_argument('--ext_time', type=str, help='Extract timestamp from visible digital clock. Black background.',
                        default=False)
    parser.add_argument('--clock_coord', type=str, help='Pixel coordinates for clock location (upper left x,y, lower right x,y)',
                        default=[225,198,413,247])
    args = parser.parse_args()
    main(args.data_dir,
         detect_time=args.ext_time,
         clock_coord=args.clock_coord)  
    

# import json
# import numpy as np
# import ast

# f=open(r'C:\Users\lahir\data\kinect_hand_data\testdata\data.json')
# d=f.read()
# data=json.loads(d)
# data=ast.literal_eval(data)

# d_list=[]
# for i in range(len(data)):
#     d=data[i]['keypoint_depths']
#     d_list.extend(d)




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