from mmpose.apis import MMPoseInferencer
import cv2

img_path='C:\\Users\\lahir\\Downloads\\hands\\Rhand_down.jpg'
img_path=r'C:\Users\lahir\Downloads\hands\original\scaled\test.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# instantiate the inferencer using the model alias
'''
configs                                                                 AUC            EPE
_________________________________________________________________________________________________
td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256
td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256
td-hm_hrnetv2-w18_dark-8xb64-210e_onehand10k-256x256                    0.572           23.96
td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256                     0.571           23.88
'''
inferencer = MMPoseInferencer('td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)

