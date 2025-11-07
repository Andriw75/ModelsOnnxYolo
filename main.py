import cv2
from nn.model_onnx import ModelOnnx
from loaders.plotter import Plotter

model = ModelOnnx(r"C:\Andriw\Deteccion de incidencias\DT_EN_DS\3) models\POSE\yolo11n-pose.onnx",0.7)
img_np = cv2.imread('77.jpg')

result = model(img_np)
img_out = Plotter().plot_results(img_np,result)
cv2.imwrite('plot.jpg',img_out)