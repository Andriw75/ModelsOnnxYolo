import numpy as np
import cv2
from nn.utils import get_pad_gain,nms_keypoints,scale_boxes,scale_points

class Class:
    def __init__(self,classes:np.ndarray):
        self.classes = classes

class Boxes:
    def __init__(self,xyxy:np.ndarray,conf:np.ndarray):
        self.xyxy = xyxy
        self.conf = conf

class Keypoints:
    def __init__(self,xy:np.ndarray,conf:np.ndarray):
        self.xy = xy
        self.conf = conf
        
class Result:
    def __init__(
                self,
                classes:Class,
                boxs:Boxes,
                kpts:Keypoints,
                ):
        self.classess = classes
        self.boxs = boxs
        self.kpts = kpts

class Before_After_Process:
    def __init__(self, new_shape=(640, 640),task='pose', scaleup=True, center=True):
        self.new_shape = new_shape
        self.scaleup = scaleup
        self.center = center
        self.ori_shape = (0,0)
        self.task = task
        
    def letter_box(self, image:np.ndarray)->np.ndarray:
        shape = image.shape[:2]  # current shape [height, width]
        self.ori_shape = shape
        
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_CUBIC)
            
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  
        return image

    def post_process(self,output:np.ndarray,box_score):

        r = Result(None,None,None)
        
        gain,pad = get_pad_gain(self.new_shape,self.ori_shape)
        output = nms_keypoints(output,box_score, 1 if self.task == 'pose' else 0)
        
        output[:,:4] = scale_boxes(output[:,:4],gain,pad)
        
        r.classess = Class(output[:,5])
        r.boxs = Boxes(output[:,:4],output[:,4])
        
        if self.task == 'pose':
            output[:,6:] = scale_points(output[:,6:],gain,pad)
            data = output[:,6:].reshape(-1,17,3)
            r.kpts = Keypoints(data[:,:,:-1],data[:,:,-1])
            
        return r
