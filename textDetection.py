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
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.'
                   )
parser.add_argument('--nms',type=float, default=0.4,
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
    croppedImage = cropBox.cropBox(args.input, Edges, 5, 2.5, angles)

    for i in range (0, len(croppedImage)):
        croppedImage[i].show()
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(croppedImage[i], config=config)
        print(text)


