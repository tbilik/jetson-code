#!/usr/bin/python3
import jetson.inference
import jetson.utils
from PIL import Image
import numpy as np
import pytesseract
import re
import sys
import os
from os import path

def ocr(imgp):
    text = pytesseract.image_to_string(imgp, config="--oem 1")
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

# location of the OCR
pytesseract.pytesseract.tesseract_cmd = "/home/tbilik/local/bin/tesseract"

# load object detection model. The traffic sign detection model uses SSD.
signDetection = jetson.inference.detectNet("ssd-mobilenet-v2",
                                           ["--model=/home/tbilik/jetson-inference/python/training/detection/ssd/models/signs/ssd-mobilenet.onnx",
                                            "--labels=/home/tbilik/jetson-inference/python/training/detection/ssd/models/signs/labels.txt",
                                            "--input-blob=input_0",
                                            "--output-cvg=scores",
                                            "--output-bbox=boxes",
                                            "--overlay=none"], threshold=0.5)

# code for loading the classification model
#imagenet = jetson.inference.imageNet("Resnet18",
#                                     ["--model=/home/tbilik/jetson-inference/python/training#/classification/models/signs/resnet18.onnx",
#                                      "--labels=/home/tbilik/jetson-inference/python/trainin#g/classification/data/signs/labels.txt",
#                                      "--input_blob=input_0",
#                                      "--output_blob=output_0"])

demoMode = False

# find image source (or activate demo mode if none provided)
if len(sys.argv) > 1:
    if sys.argv[1] == "camera":
        inp = jetson.utils.videoSource("csi://0",
                                     ["--input-width=1920","--input-height=1080"])
    else:
        inp = jetson.utils.videoSource(sys.argv[1])
else:
    demoMode = True
    # import sixel for image viewing
    import sixel
    sixelWriter = sixel.SixelWriter()
    # import readline for file tab completion
    import readline
    readline.set_completer_delims(' \t\n=')
    readline.parse_and_bind("tab: complete")

# the display_driver relies on the spelled out version of the numbers,
# as the spelled out versions were used as labels for the classification model.
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
while demoMode or inp.IsStreaming:
    if demoMode:
        # wait for child process to finish in demo mode
        try:
            os.waitpid(pid, 0)
        except:
            pass
        fileName = input("Enter image file or vehicle speed (press q to exit): ")
        if fileName == "q":
            exit()
        elif path.exists(fileName):
            inp = jetson.utils.videoSource(fileName)
            sixelWriter.draw(fileName)
        else:
            try:
                spd = int(fileName)
                with open("/home/tbilik/display_fifo","w") as fp:
                    fp.write("A" + str(spd))
            except:
                print("File not found.")
            continue
    img = inp.Capture()

    detections = signDetection.Detect(img, overlay="none")

    if len(detections) == 0:
        continue
    else:
        print(str(len(detections)) + " sign(s) detected")
        Image.fromarray(jetson.utils.cudaToNumpy(img)).save("testing/" + str(i) + ".png")
        i += 1
    
    img_grayscale = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format='gray8')

    jetson.utils.cudaConvertColor(img, img_grayscale)

    for detection in detections:
        left = int(detection.Left)
        top = int(detection.Top)
        right = int(detection.Right)
        bottom = int(detection.Bottom)

        # run a center crop to clip out sign border. Currently it's a 90% crop
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
        #Image.fromarray(im).save("testing/" + str(i) + ".jpg")
        #i += 1
        
        im = np.uint8((im>128)*255)
        if demoMode or sys.argv[1] != "camera":
            Image.fromarray(im).save("test.png")
        if demoMode:
            sixelWriter.draw("test.png")
            sys.stdout.flush()
        
        pid = os.fork()
        if pid == 0:
            ocr(im)
            os._exit(os.EX_OK)

        del temp
    del img_grayscale
