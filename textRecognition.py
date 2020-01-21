from PIL import Image
from IPython.display import display
import PIL
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

image = Image.open("Prod-Pic-cargodoor.jpg")

image = image.crop((293, 73, 337, 95))

image = image.resize((int(image.width * 3), int(image.height * 3)))
image.show()
config = ("-l eng --oem 1 --psm 7")
text = pytesseract.image_to_string(image, config=config)
print(text)