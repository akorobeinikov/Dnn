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
    classNames = {0: 'background',
                  1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                  5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                  10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                  14: 'motorbike', 15: 'person', 16: 'pottedplant',
                  17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
    while True:
        ret, img = cap.read()
        output = Net.processImage(img)
        (h, w) = img.shape[:2]
        for i in range(0, output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(output[0, 0, i, 1])  # Class label
                # Object location
                xLeftBottom = int(output[0, 0, i, 3] * 300)
                yLeftBottom = int(output[0, 0, i, 4] * 300)
                xRightTop = int(output[0, 0, i, 5] * 300)
                yRightTop = int(output[0, 0, i, 6] * 300)
                # Factor for scale to original size of image
                heightFactor = img.shape[0] / 300.0
                widthFactor = img.shape[1] / 300.0
                # Scale object detection to image
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)
                # Draw location of object
                cv2.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0), 2)
                # Draw label and confidence of prediction in image resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(img, (xLeftBottom, yLeftBottom - labelSize[1]),
                                  (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                    print(label)  # print class and confidence
        cv2.imshow("Output", img)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break