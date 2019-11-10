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
        img = cv2.imread(image)
        # forward

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