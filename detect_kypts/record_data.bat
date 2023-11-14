@ECHO off
python kinect//azure_kinect_recorder_my.py --output C:\\Users\\lahir\\data\\CPR_experiment\\test\\kinect --n -1
python cannon//remote_shoot_click.py --n -1
python arduino//readarduino.py --outpath 'C:\\Users\\lahir\\data\\CPR_experiment\\test\\arduino\\'