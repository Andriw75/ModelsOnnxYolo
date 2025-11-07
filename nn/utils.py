import numpy as np

def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    box_xyxy = box.copy()
    box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
    box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
    box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
    box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
    return box_xyxy

def nms(dets,scores,thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nms_keypoints(data:np.ndarray,conf_thres = 0.3,nc = 1):
    nc = nc or (data.shape[1] - 4)  
    mi = 4 + nc  
    xc = np.max(data[:, 4:mi], axis=1) > conf_thres
    
    data = data.transpose(0,2,1)[0]
    data = data[xc[0]]
    box, cls, mask = np.split(data, [4, 4 + nc], axis=1)
    
    conf = np.amax(cls, axis=1, keepdims=True)  # Valores máximos
    j = np.argmax(cls, axis=1, keepdims=True)
    
    data = np.concatenate(
        [box, conf, j.astype(np.float32), mask], 
        axis=1
    )
    
    scores = data[:,4]
    boxes = xywh2xyxy(data[:,:4])
    data = data[nms(boxes,scores,0.3)]
    
    data[:,:4] = xywh2xyxy(data[:,:4])
    return data

def scale_boxes(boxes,gain,pad):
    boxes[:,[0,2]] -= pad[0]
    boxes[:,[1,3]] -= pad[1]
    boxes = np.maximum(boxes, 0)
    boxes[..., :4] /= gain
    return boxes

def scale_points(points:np.ndarray,gain,pad):
    points = points.reshape(-1,17,3)
    points[:,:,0] -= pad[0]
    points[:,:,1] -= pad[1]
    points[:,:,0] /= gain
    points[:,:,1] /= gain
    return points.reshape(-1,51)

###############

def get_pad_gain(shape_new,shape_ori):
    gain = min(shape_new[0] / shape_ori[0], shape_new[1] / shape_ori[1])  # gain  = old / new
    pad = (
        round((shape_new[1] - shape_ori[1] * gain) / 2 - 0.1),
        round((shape_new[0] - shape_ori[0] * gain) / 2 - 0.1),
    )
    return gain,pad
  
