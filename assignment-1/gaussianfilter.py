from PIL import Image
import numpy as np
from scipy import signal

# read greyscale image to array and normalize the values
img = np.array(Image.open('scene.pgm').convert('L'),dtype=float)
img /= 255

width = int(raw_input("Enter width of the kernel:"))
# width = 3
# Check for odd width of kernel
while width%2 == 0:
    print "Enter odd number as width."
    width = int(raw_input("Enter width of the kernel:"))

# Build approximate gaussian kernel (blur)
blur = []
for i in range(width):
    if i <= width / 2:
        blur.append(i + 1)
    else:
        blur.append(width - i)

blurarray = np.array(blur)
norm1 = blurarray / float(blurarray.sum())

# Build 2D kernel from 1D by multiplying the 1D kernel with its transpose
filter2d = np.matrix(norm1).T * np.matrix(norm1)
print "2D gaussian kernel approximation"
print filter2d

# Do 2D convolution with valid boundary condition
grad = signal.convolve2d(img, filter2d, mode='valid')
# Un-normalize the values and convert to type integer
grad = (grad * 255).astype(np.uint8)

# Show convolved image in greyscale format
img1 = Image.fromarray(grad, 'L')
img1.save("blurred.jpg")
# img1.show()
