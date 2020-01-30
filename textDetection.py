import cv2 as cv
import argparse
import getEdges, cropBox
import pytesseract

parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
parser.add_argument('--thr',type=float, default=0.1,
                    help='Confidence threshold.'
                   )
parser.add_argument('--nms',type=float, default=0.2,
                    help='Non-maximum suppression threshold.'
                   )

args = parser.parse_args()

if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model

    cap = cv.VideoCapture(args.input if args.input else 0)
    Edges, angles = getEdges.getEdges(args.input, cap, model, inpWidth, inpHeight, confThreshold, nmsThreshold)
    croppedImage = cropBox.cropBox(args.input, Edges, 4, 2, angles)

    text = []
    meaning = []
    category = ['U', 'Z', 'J']
    info = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']
    for i in range (0, len(croppedImage)):
        #croppedImage[i].show()
        config = ("-l eng --oem 1 --psm 7")
        if pytesseract.image_to_string(croppedImage[i], config=config) != "":
            temp = pytesseract.image_to_string(croppedImage[i], config=config)
            temp = ''.join(e for e in temp if e.isalnum())
            for char in temp :
                if temp[-1] in category:
                    bool = "owner"
                if char.isalpha() is False:
                    bool = "typeCode"
                    break
            if len(temp) != 4 or temp == "TARE":
                bool = "unknown"
            if len(temp) == 6 or len(temp) == 7:
                for char in temp:
                    bool = "serial"
                    if char.isnumeric() is False:
                        bool = "unknown"
                        break
        meaning.append(bool)
        text.append(temp)
    print(text)

    try:
        info[0] = text[meaning.index("owner")][0:3]
    except ValueError:
        info[0] = "NULL"

    if info[0] != "NULL":
        info[1] = text[meaning.index("owner")][3:4]
    try:
        if len(text[meaning.index("serial")]) == 7:
            info[2] = text[meaning.index("serial")][0:6]
            info[3] = text[meaning.index("serial")][6:7]
        else:
            info[2] = text[meaning.index("serial")]
    except ValueError:
        info[2] = "NULL"
        info[3] = "NULL"

    try:
        info[4] = text[meaning.index("typeCode")]
        if info[4][2:3] == "6":
            info[4] = info[4][0:1] + info[4][1:2] + "G" + info[4][3:4]
    except ValueError:
        info[4] = "NULL"

    print(info)





