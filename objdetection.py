#Object Detection

#Imporing Libraries
import torch
from torch.autograd import Variable
import cv2 as cv
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y_output = net(x)
    detections = y_output.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            point = (detections[0, i, j, 1:] * scale).numpy()
            cv.rectangle(frame, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), (255,  0, 0), 2)
            cv.putText(frame, labelmap[i - 1], (int(point[0]), int(point[1])), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
            j += 1 
    return frame

#Creating the SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

#Creating Transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('one.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('one_output.mp4', fps = fps)
for i, frame in enumerate(reader):
    detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()






