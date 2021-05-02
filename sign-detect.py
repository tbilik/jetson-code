#!/usr/bin/python3
import jetson.inference
import jetson.utils
from PIL import Image
import numpy as np
import pytesseract
import re
import sys
import os

pytesseract.pytesseract.tesseract_cmd = "/home/tbilik/local/bin/tesseract"

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


if sys.argv[1] == "camera":
    input = jetson.utils.videoSource("csi://0",
                                     ["--input-width=1920","--input-height=1080"])
else:
    input = jetson.utils.videoSource(sys.argv[1])

signs = {
    10: "ten",
    15: "fifteen",
    20: "twenty",
    25: "twentyfive",
    30: "thirty",
    35: "thirtyfive",
    40: "forty",
    45: "fortyfive",
    50: "fifty",
    55: "fiftyfive",
    60: "sixty",
    65: "sixtyfive",
    70: "seventy",
    100: "stop"
}

# for debugging purposes
i = 0

for img in os.listdir("testing/"):
    try:
        if int(img.split(".")[0]) > i:
            i = int(img.split(".")[0])
    except:
        print("issue with image: " + img)

i += 1

#output = jetson.utils.videoOutput("test.jpg")
while input.IsStreaming:
    img = input.Capture()

    detections = signDetection.Detect(img, overlay="none")

    if len(detections) == 0:
        continue
    
    img_grayscale = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format='gray8')

    jetson.utils.cudaConvertColor(img, img_grayscale)

    for detection in detections:
        print("traffic sign detected")
        left = int(detection.Left)
        top = int(detection.Top)
        right = int(detection.Right)
        bottom = int(detection.Bottom)

        crop_factor = 0.9
        
        crop_border = (
            int((1.0 - crop_factor) * 0.5 * (right-left)),
            int((1.0 - crop_factor) * 0.5 * (bottom-top)))
        
        crop_roi = (
            left + crop_border[0],
            top + crop_border[1],
            right - crop_border[0],
            bottom - crop_border[1]
        )
        
        temp = jetson.utils.cudaAllocMapped(width=crop_roi[2]-crop_roi[0],height=crop_roi[3]-crop_roi[1],format="gray8")
        jetson.utils.cudaCrop(img_grayscale,temp,crop_roi)
        #output.Render(temp)
        #class_idx, confidence = imagenet.Classify(temp)
        jetson.utils.cudaDeviceSynchronize()
        im = jetson.utils.cudaToNumpy(temp)
        #cropped = im.crop((detection.Left, detection.Top, detection.Right, detection.Bottom))
        im = np.uint8((im.reshape(im.shape[0],im.shape[1])))

        # for testing/debugging purposes
        Image.fromarray(im).save("testing/" + str(i) + ".jpg")
        i += 1
        
        im = np.uint8((im>128)*255)
        if sys.argv[1] != "camera":
            Image.fromarray(im).save("test.jpg")
        text = pytesseract.image_to_string(im, config="--oem 1")
        print(text)
        try:
            speedLimit = 0
            if re.search("(?i)SPEED(.*)LIMIT(.*)\d\d", text, re.S) is not None:
                speedLimit = int(re.search("(?i)(.*)SPEED(.*)LIMIT(.*)\d\d", text, re.S).group()[-2:])
            elif re.search("(?i)\d\d(.*)M(.*)P(.*)H", text, re.S) is not None:
                speedLimit = int(re.search("(?i)\d\d(.*)M(.*)P(.*)H", text, re.S).group()[:2])
            elif re.search("(?i)STOP", text, re.S) is not None:
                speedLimit = 100

            if speedLimit in signs:
                with open("/home/tbilik/display_fifo","w") as fp:
                    fp.write("B" + signs[speedLimit])
                
        except:
            print("Text isn't right")
        del temp
    del img_grayscale
