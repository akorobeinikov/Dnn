import sys
sys.path.append("../src")
import argparse
import DnnAdapter as dnn

def createArgparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str,
                        default="C:\\Users\\aalle\\OneDrive\\Рабочий стол\\new programms\\Python dnn\\Dnn\\Semantic\\intel\\road-segmentation-adas-0001\\FP16\\road_seg.bin")
    parser.add_argument('-c', '--config', type=str,
                        default="C:\\Users\\aalle\\OneDrive\\Рабочий стол\\new programms\\Python dnn\\Dnn\\Semantic\\intel\\road-segmentation-adas-0001\\FP16\\road_seg.xml")
    parser.add_argument("-t", '--type', type=str, default='road_segmentation')
    parser.add_argument("-i", '--image', type=str, default='C:\\images\\road.png')
    return parser.parse_args()


# Main
if __name__ == '__main__':
    args = createArgparse()
    Net = dnn.DnnAdapter(args.weights, args.config, args.type)
    image = args.image
    Net.processImage(image)
