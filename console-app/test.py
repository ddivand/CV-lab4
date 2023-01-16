import jetson.inference
import jetson.utils
import time
import os

from transformers import pipeline
from PIL import Image

import argparse
import math
import sys
import numpy as np


load_time = time.time()
# load the object detection detection_network
detect_net_load_time_start = time.time()
detection_net = jetson.inference.detectNet("ssd-mobiledetection_net-v2", sys.argv, 0.3)
print("Detect net load time: {}".format(time.time() - detect_net_load_time_start))

image_net_load_time_start = time.time()
classifier = pipeline("image-classification", model="_insert_path_to_your_model_here_", device = 0)
print("Image net load time: {}".format(time.time() - image_net_load_time_start))

# create utils
font = jetson.utils.cudaFont()

cellphone_class_ids = [
	77
]

# analyze image and draw phone boxes
def AnalyzeImage(input_pil):
	input_array = np.array(input_pil)
	input_cuda_image = jetson.utils.cudaFromNumpy(input_array)

	detect_time = 0.0
	classify_time = 0.0

	detect_time_start = time.time()
	# detect objects in the image (with overlay)
	detections = detection_net.Detect(input_cuda_image, overlay="none")
	detect_time = time.time() - detect_time_start

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
			classify_time_start = time.time()
			result = classifier(cropped_pil_image)
			classify_time += time.time() - classify_time_start
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

	return output_pil, classify_time, detect_time

sum_detect_time = 0.0
sum_classify_time = 0.0
sum_other_time = 0.0
sum_exec_time = 0.0
	
i = 0;
j = 0;

while True:
	for filename in os.listdir('_images_dir_path_here_'):
		print(filename)

		exe_time_start = time.time()
		_, classify_time, detect_time = AnalyzeImage(Image.open('_images_dir_path_here_'+filename))
		exec_time = time.time() - exe_time_start
		other_time = exec_time - (classify_time + detect_time)

		print("Stats for {}, iteration: {}".format(filename, i))
		print("Detect time: {}".format(detect_time))
		print("Classify time: {}".format(classify_time))
		print("Util time: {}".format(other_time))
		print("Exec time: {}".format(exec_time))

		sum_detect_time += detect_time
		if classify_time != 0.0:
			j += 1
			sum_classify_time += classify_time
		sum_other_time += other_time
		sum_exec_time += exec_time

		i += 1

		print("Average detect time: {}".format(sum_detect_time/i))
		if j != 0:
			print("Average classify time: {}".format(sum_classify_time/j))
		print("Average util time: {}".format(sum_other_time/i))
		print("Average exec time: {}".format(sum_exec_time/i))
		print("------------------------------------------")


