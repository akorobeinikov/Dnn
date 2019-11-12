import sys
import cv2
sys.path.append("../src")
import argparse
import DnnAdapter as dnn

def createArgparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str,
                        default="C:\\images\\face.caffemodel")
    parser.add_argument('-c', '--config', type=str,
                        default="C:\\images\\face.prototxt")
    parser.add_argument("-t", '--type', type=str, default='face_detection')
    parser.add_argument("-i", '--image', type=str, default="")
    return parser.parse_args()

# Main
if __name__ == '__main__':
    args = createArgparse()
    Net = dnn.DnnAdapter(args.weights, args.config, args.type)
    if args.type != "face_GUI":
        image = args.image
        Net.processImage(image)

