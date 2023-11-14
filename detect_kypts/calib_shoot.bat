@ECHO off
for /L %%i in (1,1,20) do (
  python kinect//azure_kinect_recorder_my.py --output C:\\Users\\lahir\\data\\kinect_hand_data\\test\\ --n 1
  python cannon//remote_shoot_click.py --n 1
)