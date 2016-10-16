import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
from PIL import Image

# Read image and convert to grayscale
img1 = cv2.imread('scene.pgm')
img2 = cv2.imread('book.pgm')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find SIFT features and descriptors
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Draw keypoints and save each image
draw1 = cv2.drawKeypoints(img1,kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
draw2 = cv2.drawKeypoints(img2,kp2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_img1.jpg',draw1)
cv2.imwrite('sift_img2.jpg',draw2)

# Brute Force Matcher for finding 2 closest neighbor
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Find good matches with threshold between closest and second closest neighbour set to 0.9
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)

# RANSAC fitting running for N iterations
N = 100

src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

maxCount = 0
for i in range(N):
    points = random.sample(range(len(src_pts)),3)
    # Create matrices X(6x6), X_dash(6x1) to solve XA = X_dash
    X = []
    X_dash = []
    for j in range(3):
        X.append([src_pts[points[j]][0], src_pts[points[j]][1], 0, 0, 1, 0])
        X.append([0, 0, src_pts[points[j]][0], src_pts[points[j]][1], 0, 1])
        X_dash.append([dst_pts[points[j]][0]])
        X_dash.append([dst_pts[points[j]][1]])
    X = np.matrix(X)
    X_dash = np.matrix(X_dash)
    # Check for singular matrix for X and solve for A
    try:
        A = np.linalg.solve(X, X_dash)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            continue

    # Check for no of inliers (count) within a radius of 10 pixels and storing in temp
    temp = []
    count = 0
    for l in range(len(src_pts)):
        X = []
        X.append([src_pts[l][0], src_pts[l][1], 0, 0, 1, 0])
        X.append([0, 0, src_pts[l][0], src_pts[l][1], 0, 1])
        X = np.matrix(X)
        transformedPoint = np.dot(X,A)
        originalPoint = np.matrix([dst_pts[l][0], dst_pts[l][1]])
        if distance.euclidean(originalPoint, transformedPoint) < 10:
            count += 1
            temp.append(l)

    # Store the configuration with most inliers in bestA
    if count > maxCount:
        maxCount = count
        inliers = temp[:]
        bestA = A[:]

# Calculate A matrix including all inliers
transformedInliers = []
X = []
X_dash = []
for i in range(len(inliers)):
    X.append([src_pts[inliers[i]][0], src_pts[inliers[i]][1], 0, 0, 1, 0])
    X.append([0, 0, src_pts[inliers[i]][0], src_pts[inliers[i]][1], 0, 1])
    X_dash.append([dst_pts[inliers[i]][0]])
    X_dash.append([dst_pts[inliers[i]][1]])

X = np.matrix(X)
X_dash = np.matrix(X_dash)
A = np.linalg.lstsq(X, X_dash)
bestA = A[0]

# Display the 2 images side-by-side for feature matching
images = map(Image.open, ['sift_img1.jpg', 'sift_img2.jpg'])
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

combinedImage = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
    combinedImage.paste(im, (x_offset,0))
    x_offset += im.size[0]

combinedImage = np.array(combinedImage)

# Draw lines between matching features (inliers)
for i in range(len(inliers)):
    cv2.line(combinedImage, (src_pts[inliers[i]][0], src_pts[inliers[i]][1]),
             (dst_pts[inliers[i]][0] + np.float32(widths[0]), dst_pts[inliers[i]][1]), (0,0,255))

cv2.imwrite("featurematching.jpg", combinedImage)

# Construct H matrix(2X3) from bestA
bestA = np.array(bestA).ravel()
H = []
H.append([bestA[0], bestA[1], bestA[4]])
H.append([bestA[2], bestA[3], bestA[5]])
H = np.matrix(H)
print "H Matrix"
print H

# Affine transformation using the calculated H matrix
final = cv2.warpAffine(img1, H, (img1.shape[1],img1.shape[0]))
# Compare the original and transformed images
# plt.subplot(121),plt.imshow(img1),plt.title('Input')
# plt.subplot(122),plt.imshow(final),plt.title('Output')
# plt.show()
cv2.imwrite("affinetransformed.jpg", final)
