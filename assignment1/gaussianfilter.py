from PIL import Image
import numpy as np
from scipy import signal

# read image to array
im = np.array(Image.open('scene.pgm').convert('L'),dtype=float)
im /= 255
print im.shape
#width = int(raw_input("Enter width of the kernel:"))

def gaussianBlur(img, width):
    blur = []
    for i in range(width):
        if i <= width / 2:
            blur.append(i + 1)
        else:
            blur.append(width - i)

    blurarray = np.array(blur,dtype=float)
    norm1 = blurarray / blurarray.sum()

    print list(norm1)

    filter2d = np.mat(norm1).T * np.mat(norm1)
    print filter2d.shape

    grad = signal.convolve2d(img, filter2d, mode='valid')
    print grad.shape

    grad = (grad * 255).astype(np.uint8)
    img1 = Image.fromarray(grad, 'L')
    img1.save("filtered.jpg")
    img1.show()
    img = (img * 255).astype(np.uint8)
    img2 = Image.fromarray(img, 'L')
    img2.save("original.jpg")
    img2.show()
gaussianBlur(im, width=5)
