import cv2
import numpy as np

class DnnAdapter:
    def __init__(self, args: any) -> any:
        self.task_type = args["type"]
        try:
            self.weights = args["weights"]
            self.config = args["config"]
        except KeyError:
            if self.task_type == 'classification':
                self.weights = "../Classificator/classification.caffemodel"
                self.config = "../Classificator/classification.prototxt"
            if self.task_type == "face_detection" or self.task_type == "face_GUI":
                self.weights = "../Face_Detector/mobilenet-ssd.caffemodel"
                self.config = "../Face_Detector/mobilenet-ssd.prototxt"
            if self.task_type == 'road_segmentation':
                self.weights = "../Semantic/road_seg.bin"
                self.config = "../Semantic/road_seg.xml"
            if self.task_type == 'semantic_segmentation':
                self.weights = "../Semantic/seg.bin"
                self.config = "../Semantic/seg.xml"
        self.net = cv2.dnn.readNet(self.weights, self.config)
        self.scalefactor = 1.0
        self.width, self.height = (0, 0)

    def processImage(self, image):
        # Read image
        if type(image) == str:
            img = cv2.imread(image)
        else:
            img = image
        # forward
        if self.task_type == 'road_segmentation':
            self.width, self.height = (896, 512)
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (self.width, self.height)), self.scalefactor, (self.width, self.height))
            self.net.setInput(blob)
            result = self.net.forward()
            self._output_segmentation(result, img)
        if self.task_type == 'semantic_segmentation':
            self.width, self.height = (2048, 1024)
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (self.width, self.height)), self.scalefactor, (self.width, self.height))
            self.net.setInput(blob)
            result = self.net.forward()
            self._output_segmentation(result, img)
        if self.task_type == 'face_detection' or self.task_type == 'face_GUI':
            mean = (127.5, 127.5, 127.5)
            self.scalefactor = 1/127.5
            self.width, self.height = (300, 300)
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (self.width, self.height)), self.scalefactor, (self.width, self.height), mean, False)
            self.net.setInput(blob)
            result = self.net.forward()
            if self.task_type == 'face_detection':
                self._output_face_detection(result, img)
            else:
                return result
        if self.task_type == 'classification':
            mean = (104, 117, 123)
            self.width, self.height = (227, 227)
            blob = cv2.dnn.blobFromImage(img, self.scalefactor, (self.width, self.height), mean)
            self.net.setInput(blob)
            result = self.net.forward()
            self._outputClassification(result)

    def _output_segmentation(self, result, img):
        if self.task_type == 'road_segmentation':
            image = cv2.resize(img, (self.width, self.height))
            cv2.imshow("Source", image)
            flag = 0
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 0, 0)]  # BGR of BG, road, crub, mark
            for i in range(self.height):
                for j in range(self.width):
                    lst = [x[i][j] for x in result[0]]
                    index = lst.index(max(lst))
                    image[i][j] = 0.5 * image[i][j] + 0.5 * np.array(colors[index])
            cv2.imshow("Result", image)
            cv2.waitKey(0)
        else:
            image = cv2.resize(img, (self.width, self.height))
            cv2.imshow("Source", image)
            colors = result[0][0]
            lst = [(89, 66, 0), (0, 3, 153), (0, 0, 255), (57, 255, 255), (218, 153, 255), (0, 0, 0), (166, 0, 218), (255, 141, 255), (0, 255, 0), (0, 149, 196),
                   (255, 255, 0), (0, 230, 230), (188, 255, 255), (255, 5, 184), (255, 99, 71), (255, 23, 27), (0, 90, 198), (0, 2, 79), (152, 205, 0), (255, 205, 255)]
            for i in range(0, self.height):
                for j in range(0, self.width):
                    image[i][j] = 0.5 * image[i][j] + 0.5 * np.array(lst[int(colors[i][j])])
            print(result[0][0].shape)
            cv2.imshow("Result", image)
            cv2.waitKey(0)


    def _outputClassification(self, output):
        output = [x[0][0] for x in output[0]]
        _, confidence, _, points = cv2.minMaxLoc(np.array(output))
        with open('../Classificator/labels.txt') as f:
            classes = [x for x in f]
        # Top 1 classification by function minMaxLoc
        print("Class : ", points[1], " = ", classes[points[1]], " with confidence = ", confidence)
        print('\n')
        # Top 3 classification
        indexes = np.argsort(output)[-3:]
        for i in reversed(indexes):
            print('class:', classes[i], ' probability:', output[i])


    def _output_face_detection(self, detections, img):
        img_resized = cv2.resize(img, (300, 300))

        # Size of image resize (300x300)
        cols = img_resized.shape[1]
        rows = img_resized.shape[0]

        # Labels of network.
        classNames = {0: 'background',
                      1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                      5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                      10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                      14: 'motorbike', 15: 'person', 16: 'pottedplant',
                      17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

        # For get the class and location of object detected,
        # There is a fix index for class, location and confidence
        # value in detections array .
        print(detections)
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > 0.5:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label
                # Object location
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)
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
        cv2.waitKey(0)