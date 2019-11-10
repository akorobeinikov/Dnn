import sys
sys.path.append("../src")
import argparse
import DnnAdapter as dnn

def createArgparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str,
                        default="C:\\images\\two.caffemodel")
    parser.add_argument('-c', '--config', type=str,
                        default="C:\\images\\two.prototxt")
    parser.add_argument("-t", '--type', type=str, default='classification')
    parser.add_argument("-i", '--image', type=str)
    return parser.parse_args()

# Main
if __name__ == '__main__':
    args = createArgparse()
    Net = dnn.DnnAdapter(args.weights, args.config, args.type)
    image = args.image
    Net.processImage(image)
