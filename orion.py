import cv2
import numpy as np
import pyttsx3
from picamera2 import Picamera2
import time

MODEL_PATH = "/home/pi/yolov8n.onnx"
CLASSES_FILE = "/home/pi/coco_ptbr.txt"
CONF_THRESHOLD = 0.3
INPUT_SIZE = 320

engine = pyttsx3.init()
engine.setProperty('rate', 150)
try:
    engine.setProperty('voice', 'brazil')
except:
    pass

with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
    classes = [line.strip().lower() for line in f.readlines()]

net = cv2.dnn.readNetFromONNX(MODEL_PATH)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (320, 240)}))
picam2.start()

print("ORION com YOLOv8n ativo – versão nova mais precisa")

ultimo_aviso = 0
objetos_antes = set()

while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()                    # saída nova do YOLOv8-ONNX
    detections = outputs[0].transpose(1, 0)    # (8400, 84)

    objetos_agora = set()

    for det in detections:
        scores = det[4:]
        class_id = np.argmax(scores)
        conf = scores[class_id]
        if conf > CONF_THRESHOLD:
            label = classes[class_id]
            objetos_agora.add(label)

    if objetos_agora != objetos_antes or time.time() - ultimo_aviso > 5:
        if objetos_agora:
            falar = ", ".join(sorted(objetos_agora)[:4])
            texto = f"Atenção: {falar} a sua frente"
            if len(objetos_agora) > 4:
                texto += " e outros objetos"
            print(f"ORION → {texto}")
            engine.say(texto)
            engine.runAndWait()

        objetos_antes = objetos_agora.copy()
        ultimo_aviso = time.time()

    time.sleep(0.5)

picam2.stop()
print("ORION desligado.")
