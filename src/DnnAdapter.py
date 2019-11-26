import cv2
import numpy

class DnnAdapter:
    def __init__(self, args):
        self.weights = args["weights"]
        self.config = args["config"]
        self.task_type = args["type"]
        self.net = cv2.dnn.readNet(self.weights, self.config)

        self.width, self.height = args["input"].values()

    def processImage(self, image):
        # Read image
        if type(image) == str:
            img = cv2.imread(image)
        else:
            img = image
        # forward
        if self.task_type == 'road_segmentation':
            blob = cv2.dnn.blobFromImage(img, 1.0, (self.width, self.height))
            self.net.setInput(blob)
            result = self.net.forward()
            self._output_segmentation(result, img)
        if self.task_type == 'semantic_segmentation':
            blob = cv2.dnn.blobFromImage(img, 1.0, (self.width, self.height))
            self.net.setInput(blob)
            result = self.net.forward()
            self._output_segmentation(result, img)
        if self.task_type == 'face_detection' or self.task_type == 'face_GUI':
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (self.width, self.height)), 1.0, (self.width, self.height))
            self.net.setInput(blob)
            result = self.net.forward()
            if self.task_type == 'face_detection':
                self._output_face_detection(result, img)
            else:
                return result
        if self.task_type == 'classification':
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (self.width, self.height)), 1, (self.width, self.height))
            self.net.setInput(blob)
            result = self.net.forward()
            self._outputClassification(result)

    def _output_segmentation(self, result, img):
        if self.task_type == 'road_segmentation':
            image = cv2.resize(img, (self.width, self.height))
            cv2.imshow("Output", image)
            flag = 0
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 0, 0)]  # BGR of BG, road, crub, mark
            for i in range(self.height):
                for j in range(self.width):
                    lst = [x[i][j] for x in result[0]]
                    index = lst.index(max(lst))
                    image[i][j] = colors[index]
            cv2.imshow("Result", image)
            cv2.waitKey(0)
        else:
            image = cv2.resize(img, (self.width, self.height))
            cv2.imshow("Output1", image)
            colors = result[0][0]
            lst = [(89, 66, 0), (0, 3, 153), (0, 0, 255), (99, 126, 255), (218, 153, 255), (0, 0, 0), (166, 0, 218), (255, 141, 255), (0, 255, 0), (0, 149, 196),
                   (255, 255, 0), (0, 230, 230), (188, 255, 255), (255, 5, 184), (255, 99, 71), (0, 102, 255), (0, 90, 198), (0, 2, 79), (152, 205, 0), (255, 205, 255)]
            for i in range(0, self.height):
                for j in range(0, self.width):
                    image[i][j] = lst[int(colors[i][j])]
            print(result[0][0].shape)
            cv2.imshow("Output", image)
            cv2.waitKey(0)


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
        # print(output)
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