import numpy as np
import argparse
import onnx
import keras2onnx
from keras.models import load_model
# Create, compile and train model...
def createArgparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-im", "--input", required=True,
                      type=str)
    parser.add_argument("-om", "--output",
                      required=True, type=str)
    parser.add_argument("-n", "--name", type=str, default="my_model.pb")

    return parser.parse_args()

args = createArgparse()
print(args.input)




# load keras model
model = load_model(str(args.input))

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = 'model.onnx'
onnx.save_model(onnx_model, temp_model_file)