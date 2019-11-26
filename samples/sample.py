import json
import cv2
import numpy as np
import sys
sys.path.append("../src")
import DnnAdapter as dnn


with open('model.json') as f:
    model = json.load(f)

Net = dnn.DnnAdapter(model)

if model["type"] != "face_GUI":
    Net.processImage(model["image"])
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    while (True):
        ret, img = cap.read()
        output = Net.processImage(img)
        (h, w) = img.shape[:2]
        for i in range(0, output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence > 0.5:
                box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imshow("Output", img)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break