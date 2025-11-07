from pathlib import Path
from abc import ABC,abstractmethod
import numpy as np
from nn.process import Before_After_Process

class Model(ABC):
    def __init__(self,path_model:str,score:float):
        self.path_model = Path(path_model)
        self.score = score
        
        #############
        self.task = ''
        self.size = (0,0)
        self.nm_dict = {}
        #############
        self.init_model()
        self.process = Before_After_Process(self.size,self.task)
        
    @abstractmethod
    def run_predict(self,img_np):
        pass
    
    @abstractmethod
    def init_model(self):
        pass
    
    def __call__(self,img_np:np.ndarray):
        img_np = self.process.letter_box(img_np)
        img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
        img_np = img_np.transpose((0, 3, 1, 2))
        img_np /= 255
        output = self.run_predict(img_np)
        return self.process.post_process(output,self.score)
    