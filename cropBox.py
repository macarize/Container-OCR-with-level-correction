from PIL import Image

def cropBox(input, Edges, padding, enhance, angles):
    paddingNeg = -padding
    padding = [paddingNeg, paddingNeg, padding, padding]
    croppedImage = []
    for i in range(0, len(Edges)):
        image = Image.open(input)
        center = [(Edges[i][0] + Edges[i][2])/2, (Edges[i][1] + Edges[i][3])/2]
        Edges[i] = Edges[i] + padding
        image = image.rotate(angles[i], center=center, resample=Image.BICUBIC)
        image = image.crop(tuple(Edges[i]))
        image = image.resize((int(image.width * enhance), int(image.height * enhance)))
        #image.show()

        croppedImage.append(image)
    return croppedImage

