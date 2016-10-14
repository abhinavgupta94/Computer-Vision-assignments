from PIL import Image
import numpy as np
from scipy import signal

# read image to array
img = np.array(Image.open('scene.pgm').convert('L'),dtype=float)
img /= 255
width = int(raw_input("Enter width of the kernel:"))
while width%2==0:
    print "Enter odd number as width"
    width = int(raw_input("Enter width of the kernel:"))

blur = []
for i in range(width):
    if i <= width / 2:
        blur.append((i + 1))
    else:
        blur.append((width - i))


blurarray = np.array(blur,dtype=float)
norm1 = blurarray / blurarray.sum()

print norm1

filter2d = np.matrix(norm1).T * np.matrix(norm1)

grad = signal.convolve2d(img, filter2d, mode='valid')
grad = (grad * 255).astype(np.uint8)

img1 = Image.fromarray(grad, 'L')
img1.save("filtered.jpg")
img1.show()
# img = (img * 255).astype(np.uint8)
# img2 = Image.fromarray(img, 'L')
# img2.save("original.jpg")
# img2.show()
