from mmpose.apis import MMPoseInferencer

class PredictKypt:
    '''
    model_name                                                              AUC            EPE
    _________________________________________________________________________________________________
    td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256
    td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256
    td-hm_hrnetv2-w18_dark-8xb64-210e_onehand10k-256x256                    0.572           23.96
    td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256                     0.571           23.88
    '''
    def __init__(self,model_name='td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256'):
        self.model_name=model_name
        self.inferencer = MMPoseInferencer(model_name)

    def show_kypts(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        keypoints=self.result['keypoints']
        x=[k[0] for k in keypoints]
        y=[k[1] for k in keypoints]
        image = mpimg.imread(self.img_path)
        plt.imshow(image)
        plt.scatter(x,y)
        plt.show()
    
    def get_kypts(self,img_path):
        self.img_path=img_path
        result_generator = self.inferencer(img_path, show=False)
        result = next(result_generator)
        self.result=result['predictions'][0][0]
        return self.result 



# #how to use this class
# img_path=r'C:\Users\lahir\data\kinect_hand_data\frames\color\00020.jpg'        
# kp=PredictKypt()
# pred=kp.get_kypts(img_path)
# #get keypoints from the prediction result
# keypoints=pred['keypoints']
# #show the hand image with keypoints
# kp.show_kypts()




