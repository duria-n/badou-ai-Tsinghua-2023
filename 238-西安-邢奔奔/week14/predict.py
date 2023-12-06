from keras.layers import Input
from frcnn import Frcnn
from PIL import Image

frcnn = Frcnn()

# while True:
#     img = input('img/street.jpg')
#     try:
image = Image.open('img/street.jpg')
    # except:
    #     print('Open Error! Try again!')
    #     continue
    # else:
r_image = frcnn.detect_image(image)
r_image.show()
frcnn.close_session()

