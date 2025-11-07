import cv2
import time
from threading import Thread,Lock

class RTSPStreamer:
    """
    Puede tomar una ruta rtsp o una ruta de video
    """
    def __init__(self, rtsp_url, reconnect_timeout=5, vid_stride=1):
        self.rtsp_url = rtsp_url
        self.reconnect_timeout = reconnect_timeout
        self.vid_stride = max(1, vid_stride)
        self.latest_frame = None
        self.lock = Lock()
        self.running = False
        self.thread = None
        self.cap = None
        
        self.start()

    def start(self):
        """Inicia el hilo de captura de frames"""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._capture_frames, daemon=True)
            self.thread.start()

    def _capture_frames(self):
        """Hilo principal de captura de frames con reconexión automática y vid_stride"""
        while self.running:
            try:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if not self.cap.isOpened():
                    raise RuntimeError("No se pudo conectar al stream RTSP")

                frame_count = 0
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if frame_count % self.vid_stride == 0:
                        with self.lock:
                            self.latest_frame = frame.copy()
                    
            except Exception as e:
                print(f"Error: {e}. Reintentando en {self.reconnect_timeout} segundos...")
            
            finally:
                if self.cap:
                    self.cap.release()
                if self.running:
                    time.sleep(self.reconnect_timeout)

    def get_frame(self):
        """Obtiene el frame más reciente de forma segura"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        """Detiene la captura de forma segura"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def __del__(self):
        self.stop()

import math

class LoadVideo:
    """
    Clase para iterar sobre los frames de un video o ir directamente a un frame específico.
    """

    def __init__(self, path: str, stride: int = 1, channels: int = 3, start_frame: int = 0):
        self.path = path
        self.stride = stride
        self.channels = channels
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {path}")

        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Validar y saltar al frame inicial
        if start_frame < 0 or start_frame >= self.frames:
            raise ValueError(f"start_frame {start_frame} fuera de rango (0-{self.frames-1})")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.frame_num = start_frame

        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(self.stride):
            success = self.cap.grab()
            if not success:
                self.cap.release()
                raise StopIteration

        success, frame = self.cap.retrieve()
        if not success:
            self.cap.release()
            raise StopIteration

        self.frame_num += self.stride

        if self.channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None]

        return self.frame_num, frame

    def __len__(self):
        # Ajustamos la longitud según el frame inicial
        return math.ceil((self.frames - self.frame_num) / self.stride)
    
    def seek(self, frame_number: int):
        """
        Salta directamente al frame especificado y lo devuelve.
        """
        if frame_number < 0 or frame_number >= self.frames:
            raise ValueError(f"Frame {frame_number} fuera de rango (0-{self.frames-1})")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = self.cap.read()
        if not success:
            raise RuntimeError(f"No se pudo leer el frame {frame_number}")

        self.frame_num = frame_number

        if self.channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None]

        return frame_number, frame


if __name__ == "__main__":
    # ===== Ejemplo con video local =====
    
    video_path = "RUTA/A/TU/VIDEO.mp4"
    loader = LoadVideo(video_path, stride=7, start_frame=50)
    print(f"Total frames a iterar: {len(loader)}")

    for n, frame in loader:
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()


    # ===== Ejemplo con RTSP =====
    rtsp_url = "rtsp://USUARIO:CONTRASEÑA@IP_CAMARA"
    stream = RTSPStreamer(rtsp_url, vid_stride=1)

    try:
        while True:
            frame = stream.get_frame()
            if frame is not None:
                cv2.imshow('RTSP Stream', frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC para salir
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
