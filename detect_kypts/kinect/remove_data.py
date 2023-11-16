import json
import numpy as np
import ast
import matplotlib.pyplot as plt
import argparse
import os

def parse_ranges(input_string):
    result = []
    ranges = input_string.split(',')
    
    for r in ranges:
        if '-' in r:
            start, end = map(int, r.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(r))
    
    return result


def main(args):
    f=open(args.datafile)
    d=f.read()
    data=json.loads(d)
    data=ast.literal_eval(data)

    #keep only these img_nums from the list (remove errorneously detected hands)
    to_keep=parse_ranges(args.tokeep)
    new_data={}
    ts_list=[]
    ind_list=[]
    for index, (key, value) in enumerate(data.items()):
        img_num=value['img_num']
        if img_num in to_keep:
            new_data[index]=value
            ts_list.append(new_data[index]['ts'])
            ind_list.append(index)
    #sort the dict based on ts
    ts_list=np.array(ts_list)
    sort_args=np.argsort(ts_list)
    sort_inds=np.array(ind_list)[sort_args]

    sorted_data={}
    for i,ind in enumerate(sort_inds):
        sorted_data[i]=new_data[ind]

    parent_dir=os.path.dirname(args.datafile)
    json_file=os.path.join(parent_dir,'sorted_data.json')
    with open(json_file, "w") as outfile: 
        json.dump(str(sorted_data), outfile) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect hand keypoints')
    parser.add_argument('--datafile', type=str, help='directory containing data',
                        default=r'C:\Users\lahir\data\CPR_experiment\test\kinect\data.json')
    parser.add_argument('--tokeep', type=str, nargs='+', help='valid range of images e.g. 1-6,15-35,234-567 (inclusive)',
                        default='243-519')
    args = parser.parse_args()
    main(args) 




# f=open(r'C:\Users\lahir\data\CPR_experiment\test\kinect\sorted_data.json')
# d=f.read()
# data=json.loads(d)
# data=ast.literal_eval(data)

# #keep only these img_nums from the list (remove errorneously detected hands)
# ts_list=[]
# for index, (key, value) in enumerate(data.items()):
#     ts_list.append(data[key]['ts'])








