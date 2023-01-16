import jetson.inference
import jetson.utils

from transformers import pipeline
from PIL import Image

import argparse
import math
import sys
import numpy as np

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--detection_network", type=str, default="ssd-mobiledetection_net-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection detection_network
detection_net = jetson.inference.detectNet(opt.detection_network, sys.argv, opt.threshold)
classifier = pipeline("image-classification", model="_insert_path_to_your_model_here_")

# create utils
font = jetson.utils.cudaFont()

cellphone_class_ids = [
	77
]

# analyze image and draw phone boxes
def AnalyzeImage(input_pil):
	input_array = np.array(input_pil)
	input_cuda_image = jetson.utils.cudaFromNumpy(input_array)

	# detect objects in the image (with overlay)
	detections = detection_net.Detect(input_cuda_image, overlay="none")

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		if detection.ClassID in cellphone_class_ids:
			# Crop the image to use it in 
			roi = (
					math.floor(detection.Left),
					math.floor(detection.Top),
					math.ceil(detection.Right),
					math.ceil(detection.Bottom))
			
			cropped_cuda_image = jetson.utils.cudaAllocMapped(
					width=roi[2] - roi[0],
                    height=roi[3] - roi[1],
                    format=input_cuda_image.format)

			jetson.utils.cudaCrop(input_cuda_image, cropped_cuda_image, roi)

			cropped_array = jetson.utils.cudaToNumpy(cropped_cuda_image)

			cropped_pil_image = Image.fromarray(cropped_array, 'RGB')

			# Сlassify the image
			result = classifier(cropped_pil_image)
			confidence = result[0]['score']
			label = result[0]['label']

			# Overlay the bouding box
			jetson.utils.cudaDrawRect(
			 		input_cuda_image,
			 		(roi[0], roi[1], roi[2], roi[3]),
			 		(0, 255, 0, 100))	

			# Overlay the text label
			font.OverlayText(
			 		input_cuda_image,
			 		roi[2] - roi[0],
			 		roi[3] - roi[1],
			 		"{:05.2f}% {:s}".format(confidence * 100, label),
			 		roi[0] + 5,
			 		roi[1] + 5,
			 		font.White,
			 		font.Gray40)

	output_array = jetson.utils.cudaToNumpy(input_cuda_image)
	output_pil = Image.fromarray(output_array, 'RGB')

	return output_pil

AnalyzeImage(Image.open(opt.input_URI)).save(opt.output_URI)