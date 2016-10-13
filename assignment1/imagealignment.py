import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy
import random
from PIL import Image


img1 = cv2.imread('scene.pgm')
img2 = cv2.imread('book.pgm')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

draw1 = cv2.drawKeypoints(img1,kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
draw2 = cv2.drawKeypoints(img2,kp2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_img1.jpg',draw1)
cv2.imwrite('sift_img2.jpg',draw2)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)

N = 100

src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

maxCount = 0
for i in range(N):
    points = random.sample(range(len(src_pts)),3)

    X = []
    X_dash = []
    for j in range(3):
        X.append([src_pts[points[j]][0], src_pts[points[j]][1], 0, 0, 1, 0])
        X.append([0, 0, src_pts[points[j]][0], src_pts[points[j]][1], 0, 1])
        X_dash.append([dst_pts[points[j]][0]])
        X_dash.append([dst_pts[points[j]][1]])
    X = np.matrix(X)
    X_dash = np.matrix(X_dash)
    try:
        A = np.linalg.solve(X, X_dash)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            continue

    temp = []
    count = 0
    for l in range(len(src_pts)):
        X = []
        X.append([src_pts[l][0], src_pts[l][1], 0, 0, 1, 0])
        X.append([0, 0, src_pts[l][0], src_pts[l][1], 0, 1])
        # X = np.matrix(X)
        transformedPoint = np.dot(X,A)
        originalPoint = np.matrix([dst_pts[l][0], dst_pts[l][1]])
        if distance.euclidean(originalPoint, transformedPoint) < 10:
            count += 1
            temp.append(l)

    if count > maxCount:
        maxCount = count
        inliers = temp[:]
        bestA = A[:]

# print maxCount

transformedInliers = []
X = []
X_dash = []
for i in range(len(inliers)):
    X.append([src_pts[inliers[j]][0], src_pts[inliers[j]][1], 0, 0, 1, 0])
    X.append([0, 0, src_pts[inliers[j]][0], src_pts[inliers[j]][1], 0, 1])
    X_dash.append([dst_pts[inliers[j]][0]])
    X_dash.append([dst_pts[inliers[j]][1]])

    # transformedPoint = np.dot(X,bestA)
    # transformedInliers.extend((transformedPoint[0], transformedPoint[1]))

X = np.matrix(X)
X_dash = np.matrix(X_dash)

# X_T = np.transpose(X)
# A = np.dot(np.dot(np.dot(X_T,X),X_T),X_dash)
# A = np.linalg.solve(X, X_dash)
# print bestA

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

for i in range(len(inliers)):
    cv2.line(combinedImage, (src_pts[inliers[i]][0], src_pts[inliers[i]][1]),
             (dst_pts[inliers[i]][0] + np.float32(widths[0]), dst_pts[inliers[i]][1]), (0,0,255))

cv2.imwrite("transformed.jpg", combinedImage)

# pts = np.float32([kp1[m].pt for m in range(len(kp1))])

bestA = np.array(bestA).ravel()
H = []
H.append([bestA[0], bestA[1], bestA[4]])
H.append([bestA[2], bestA[3], bestA[5]])
H = np.matrix(H)
print H
final = cv2.warpAffine(img1, H, (img1.shape[1],img1.shape[0]))
# plt.subplot(121),plt.imshow(img1),plt.title('Input')
# plt.subplot(122),plt.imshow(final),plt.title('Output')
# plt.show()
cv2.imwrite("final.jpg", final)
