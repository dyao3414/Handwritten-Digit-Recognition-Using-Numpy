from predict import *
from train_network import forward_propagation
import numpy as np
from PIL import Image

# sample usage:
# input an image path or dir
if __name__=='__main__':
    # samlpe input
    input_path=r'C:\Users\AndyYao\Desktop\handwrite-9.png' # samlpe input, where you save the handwritten digit file
    img=Image.open(input_path)
    gray_scaled_img = img.convert('L') # convert three channels RGB to one channel gray scale image, align with training images
    gray_scaled_img = Image.eval(gray_scaled_img, lambda x: 255 - x) # revert the black and white 
    img_resized = gray_scaled_img.resize((28, 28)) # resize the input image file to 28*28 to conform the format of training images
    output_path=r'C:\Users\AndyYao\Desktop\handwrite-9-resized.png'
    img_resized.save(output_path) #sample location, change to your desired location
    print(predict(output_path))
