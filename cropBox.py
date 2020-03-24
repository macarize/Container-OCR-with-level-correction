
from PIL import Image
import cv2 as cv
import numpy as np

def cropBox(input, Edges, padding, enhance, angles):
    paddingNeg = -padding
    padding = [paddingNeg, paddingNeg, padding, padding]
    croppedImage = []
    for i in range(0, len(Edges)):
        image = Image.open(input)
        imagecv = np.array(image)
        imagecv = imagecv[:, :, ::-1].copy()
        imagecv = cv.bitwise_not(imagecv)
        imagecv = cv.cvtColor(imagecv, cv.COLOR_BGR2GRAY)
        #imagecv = cv.medianBlur(imagecv,5)
        #imagecv = cv.adaptiveThreshold(imagecv,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
         #   cv.THRESH_BINARY,11,2)
        ret, imagecv = cv.threshold(imagecv,255,255,cv.THRESH_TRUNC)
        center = [(Edges[i][0] + Edges[i][2])/2, (Edges[i][1] + Edges[i][3])/2]
        Edges[i] = Edges[i] + padding
        #imagecv = cv.cvtColor(imagecv, cv.COLOR_BGR2RGB)
        image = Image.fromarray(imagecv)
        image = image.rotate(angles[i]*0.7, center=center, resample=Image.BICUBIC) #level correction
        image = image.crop(tuple(Edges[i]))
        image = image.resize((int(image.width * enhance), int(image.height * enhance)))
        image.show()
        croppedImage.append(image)
    return croppedImage

