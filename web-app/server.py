import sys
import math
import numpy as np
import base64

import jetson.inference
import jetson.utils

from transformers import pipeline

from PIL import Image
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse

from io import BytesIO
import uvicorn, asyncio, requests #aiohttp


cellphone_class_ids = [
	77
]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"))
templates = Jinja2Templates(directory="templates")

detection_net = None
classifier = None
# font = None

# analyze image and draw phone boxes
def AnalyzeImage(input_pil):
	input_array = np.array(input_pil)
	input_cuda_image = jetson.utils.cudaFromNumpy(input_array)

	# detect objects in the image (with overlay)
	detections = detection_net.Detect(input_cuda_image, overlay="none")

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)
		print(detection_net.GetClassDesc(detection.ClassID))
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
			print(result)
			confidence = result[0]['score']
			label = result[0]['label']

			# Overlay the bouding box
			jetson.utils.cudaDrawRect(
			 		input_cuda_image,
			 		(roi[0], roi[1], roi[2], roi[3]),
			 		(0, 255, 0, 100))	
			font = jetson.utils.cudaFont()
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
	output_pil.save('./res.jpg')
	return output_pil


@app.on_event('startup')
async def setup_learner():
    global detection_net
    global classifier
    
    # global font
    print('Setup')

    # load the object detection detection_network
    detection_net = jetson.inference.detectNet("ssd-mobiledetection_net-v2", sys.argv, 0.3) #opt.detection_network, sys.argv, opt.threshold)
    classifier = pipeline("image-classification", model="/home/nano7/l4/checkpoint", device=0)

    # create utils
    # font = jetson.utils.cudaFont()
        
    print('Finish setting up nets')


@app.get('/')
async def root(request: Request):
    return templates.TemplateResponse("index.html", {'request':request})


@app.post('/analyze', response_class=PlainTextResponse)
async def analyze(request: Request):
    print("analyze")
    data = await request.form()
    if data['file'] == 'undefined':
        return {'result': 'undefined'}

    img_bytes = await (data['file'].read())
    img = Image.open(BytesIO(img_bytes))
    res = AnalyzeImage(img)
    # res.save('./res.jpg')
#    res = img

    buffer = BytesIO()
    res.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue())
    return img_str


if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5555)


