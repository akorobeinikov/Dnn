import cv2
import numpy

class DnnAdapter:
    def __init__(self, weightsPath=None, configPath=None,
                 task_type=None):
        self.weights = weightsPath
        self.config = configPath
        self.task_type = task_type
        # Create net
        self.net = cv2.dnn.readNet(self.weights, self.config)

    def processImage(self, image):
        # Read image
        if type(image) == str:
            img = cv2.imread(image)
        else:
            img = image
        # forward
        if self.task_type == 'face_detection' or self.task_type == 'face_GUI':
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (128, 128, 128))
            self.net.setInput(blob)
            result = self.net.forward()
            self._output_face_detection(result, img)
        if self.task_type == 'classification':
            blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (128, 128, 128))
            self.net.setInput(blob)
            result = self.net.forward()
            self._outputClassification(result)


    def _outputClassification(self, output):
        minval, confidence, minloc, Point = cv2.minMaxLoc(output)
        with open('../Classificator/labels.txt') as f:
            classes = [x for x in f]
        print("Class : ", Point[0], " = ", classes[Point[0]], " with confidence = ", confidence)
        print('\n')
        indexes = numpy.argsort(output[0])[-3:]
        for i in reversed(indexes):
            print('class:', classes[i], ' probability:', output[0][i])


    def _output_face_detection(self, output, img):
        (h, w) = img.shape[:2]
        print(output)
        for i in range(0, output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence > 0.5:
                box = output[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imshow("Output", img)
        cv2.waitKey(0)