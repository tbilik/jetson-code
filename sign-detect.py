import jetson.inference
import jetson.utils
from PIL import Image
import numpy as np
import pytesseract

signDetection = jetson.inference.detectNet("ssd-mobilenet-v2",
                                           ["--model=/home/tbilik/jetson-inference/python/training/detection/ssd/models/signs/ssd-mobilenet.onnx",
                                            "--labels=/home/tbilik/jetson-inference/python/training/detection/ssd/models/signs/labels.txt",
                                            "--input-blob=input_0",
                                            "--output-cvg=scores",
                                            "--output-bbox=boxes",
                                            "--overlay=none"], threshold=0.5)

#imagenet = jetson.inference.imageNet("Resnet18",
#                                     ["--model=/home/tbilik/jetson-inference/python/training#/classification/models/signs/resnet18.onnx",
#                                      "--labels=/home/tbilik/jetson-inference/python/trainin#g/classification/data/signs/labels.txt",
#                                      "--input_blob=input_0",
#                                      "--output_blob=output_0"])

input = jetson.utils.videoSource("~/speed-limit-signs/bigstock-Three-Road-Signs-119503835.jpg")
#output = jetson.utils.videoOutput("test.jpg")
while input.IsStreaming:
    img = input.Capture()

    detections = signDetection.Detect(img, overlay="none")

    if len(detections) == 0:
        continue
    
    img_grayscale = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format='gray8')

    jetson.utils.cudaConvertColor(img, img_grayscale)

    for detection in detections:
        left = int(detection.Left)
        top = int(detection.Top)
        right = int(detection.Right)
        bottom = int(detection.Bottom)
    
        temp = jetson.utils.cudaAllocMapped(width=right-left,height=bottom-top,format="gray8")
        jetson.utils.cudaCrop(img_grayscale,temp,(left,top,right,bottom))
        #output.Render(temp)
        #class_idx, confidence = imagenet.Classify(temp)
        jetson.utils.cudaDeviceSynchronize()
        im = jetson.utils.cudaToNumpy(temp)
        #cropped = im.crop((detection.Left, detection.Top, detection.Right, detection.Bottom))
        im = np.uint8((im.reshape(im.shape[0],im.shape[1])))
        text = pytesseract.image_to_string(Image.fromarray(im))
        try:
            speedLimit = int(re.match("(.*)SPEED(.*)LIMIT(.*)\d\d", text, re.S).group()[-2:])
        except:
            print("Text isn't right")
        del temp
    del img_grayscale
