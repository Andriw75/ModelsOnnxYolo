import numpy as np
import cv2

class Plotter:
    def __init__(self):
        self.skeleton = [
                [16, 14, '#00ff13'], [14, 12, '#00ff13'], [17, 15, '#00ff13'], 
                [15, 13, '#00ff13'], [12, 13, '#ff0000'], [6, 12, '#ff0000'],
                [7, 13, '#ff0000'], [6, 7, '#e4ff00'], [6, 8, '#e4ff00'], 
                [7, 9, '#e4ff00'], [8, 10, '#e4ff00'], [9, 11, '#e4ff00'], 
                [2, 3, '#00f0ff'], [1, 2, '#00f0ff'], [1, 3, '#00f0ff'], 
                [2, 4, '#00f0ff'], [3, 5, '#00f0ff'], [4, 6, '#00f0ff'], 
                [5, 7, '#00f0ff']
            ]
        
    def _hex2rgb(self, hex_color: str) -> tuple:
        """Convierte un color hexadecimal a formato BGR."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    
    def draw_lines(self, kpts: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Dibuja los puntos clave y conexiones esqueléticas en una imagen."""
        frame = frame.copy()
        s2, s1, _ = frame.shape

        for sk in self.skeleton:
            pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
            pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))

            if pos1[0] >= s1 or pos1[1] >= s2 or pos1[0] <= 0 or pos1[1] <= 0:
                continue
            if pos2[0] >= s1 or pos2[1] >= s2 or pos2[0] <= 0 or pos2[1] <= 0:
                continue

            cv2.line(frame, pos1, pos2, self._hex2rgb(sk[2]), 2)

        return frame
    
    def draw_rectangle(self,points, image):
        """
        Dibuja un rectángulo en la imagen usando las coordenadas de los puntos [x1, y1, x2, y2].
        
        :param image: La imagen sobre la que dibujar.
        :param points: Lista de 4 elementos [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = points
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return image

    