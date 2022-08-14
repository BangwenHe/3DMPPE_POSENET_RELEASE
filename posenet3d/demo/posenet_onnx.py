import cv2
import numpy as np
from onnx import numpy_helper
import onnx
import os
from PIL import Image
import onnxruntime as rt
from scipy import special
import colorsys
import random
import argparse
import sys
import time


if __name__ == "__main__":
    sess = rt.InferenceSession("posenet.onnx")
    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))

    inputs = sess.get_inputs()
    input_names = list(map(lambda input: input.name, inputs))

    outputs = sess.run(output_names, )