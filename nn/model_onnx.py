import ast
import onnxruntime as ort 
import numpy as np
from nn.model import Model

class ModelOnnx(Model):
    def __init__(self, path_model):
        super().__init__(path_model)

    def init_model(self):
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider']
            self.is_gpu = True
        self.sesion = ort.InferenceSession(str(self.path_model),providers=providers)
        
        metadata = self.sesion.get_modelmeta().custom_metadata_map
        
        self.task = metadata['task']
        self.input_name = self.sesion.get_inputs()[0].name
        self.size = ast.literal_eval(metadata['imgsz'])
        self.nm_dict = ast.literal_eval(metadata['names'])
        
        if self.sesion.get_inputs()[0].type == "tensor(float16)":
            self.input_type = np.float16
            
    def run_predict(self, img_np):
        return self.sesion.run(None,{self.input_name:img_np})[0]
