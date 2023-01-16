import jetson.inference
import jetson.utils

from transformers import pipeline
from PIL import Image

import argparse
import math
import sys

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


# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
test_output = jetson.utils.videoOutput("test_output.jpg", argv=sys.argv)

# load the object detection detection_network
detection_net = jetson.inference.detectNet(opt.detection_network, sys.argv, opt.threshold)
classifier = pipeline("image-classification", model="/home/nano7/l4/checkpoint", device = 0)

# create utils
font = jetson.utils.cudaFont()

cellphone_class_ids = [
	77
]

def AnalyzeImage(pilImage):
	return


# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = detection_net.Detect(img, overlay="none")

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)
		if detection.ClassID in cellphone_class_ids:
			# Crop the image to use it in 
			roi = (
					math.floor(detection.Left),
					math.floor(detection.Top),
					math.ceil(detection.Right),
					math.ceil(detection.Bottom))
			
			cropped_img = jetson.utils.cudaAllocMapped(
					width=roi[2] - roi[0],
                    height=roi[3] - roi[1],
                    format=img.format)

			jetson.utils.cudaCrop(img, cropped_img, roi)

			cropped_image_array = jetson.utils.cudaToNumpy(cropped_img)

			cropped_pil_image = Image.fromarray(cropped_image_array, 'RGB')

			# Сlassify the image
			result = classifier(cropped_pil_image)
			confidence = result[0]['score']
			label = result[0]['label']

			# Overlay the bouding box
			jetson.utils.cudaDrawRect(
			 		img,
			 		(roi[0], roi[1], roi[2], roi[3]),
			 		(0, 255, 0, 100))	

			# Overlay the text label
			font.OverlayText(
			 		img,
			 		roi[2] - roi[0],
			 		roi[3] - roi[1],
			 		"{:05.2f}% {:s}".format(confidence * 100, label),
			 		roi[0] + 5,
			 		roi[1] + 5,
			 		font.White,
			 		font.Gray40)

	# render the image
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.detection_network, detection_net.GetNetworkFPS()))

	# print out performance info
	detection_net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break