from mmpose.apis import MMPoseInferencer

img_path = 'C:\\Users\\lahir\\mmpose\\tests\\data\\coco\\000000000785.jpg'   # replace this with your own image path
img_path='C:\\Users\\lahir\\Downloads\\hands\\Rhand_down.jpg'

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('hand')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)