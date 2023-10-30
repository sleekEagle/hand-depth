import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from KyptPred import PredictKypt
import json
import argparse

# fps: camera FPS used
def main(data_dir,fps=5):
    rgb_dir=os.path.join(data_dir,'frames','color')
    depth_dir=os.path.join(data_dir,'frames','depth')


    #extract start timestamp
    rec_name=[file for file in os.listdir(data_dir) if file[-3:]=='mkv']
    if len(rec_name)>0:
        start_ts=float(rec_name[0].split('_')[-1][:-4])


    #if bbox_score > bb_thresh no hands are detected
    bb_thresh=0.6
    Kypt=PredictKypt()
    data={}
    rgb_files=[file for file in os.listdir(rgb_dir) if file.split('.')[-1]=='jpg']
    n=0
    for file in rgb_files:
        result=Kypt.get_kypts(os.path.join(rgb_dir,file))
        bb_score=result['bbox_score']
        img_num=int(file.split('.')[0])
        img_ts=start_ts+1/fps*img_num
        if bb_score<bb_thresh:
            #hand detected
            hand_kypts=result['keypoints']
            bb=result['bbox']
            this_data={'img_num': img_num,
                    'keypoints': result['keypoints'],
                    'bbox': result['bbox'],
                    'keypoint_scores': result['keypoint_scores'],
                    'bbox_score': result['bbox_score']}
            data[n]=this_data
            n+=1
    json_file=os.path.join(data_dir,'data.json')
    with open(json_file, "w") as outfile: 
        json.dump(str(data), outfile) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect hand keypoints')
    parser.add_argument('--data_dir', type=str, help='directory containing data',
                        default='C:\\Users\\lahir\\data\\kinect_hand_data\\CPR_data\\kinect\\session1\\')
    parser.add_argument('--ext_time', type=str, help='Extract timestamp from visible digital clock. Black background.',
                        default=True)
    parser.add_argument('--clock_coord', type=str, help='Pixel coordinates for clock location (upper left x,y, lower right x,y)',
                        default=[372,198,526,298])
    args = parser.parse_args()
    main(args.data_dir)  


