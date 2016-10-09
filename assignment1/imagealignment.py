import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
from PIL import Image


img1 = cv2.imread('scene.pgm')
img2 = cv2.imread('book.pgm')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# print len(kp1),len(kp2)
# print des1.shape

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

# print len(good)
N = 100

src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
# print src_pts[0],src_pts[1],src_pts[2]
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
# print dst_pts[0], dst_pts[1], dst_pts[2]
# print src_pts.shape, dst_pts.shape

maxCount = 0
for i in range(N):
    points = random.sample(range(len(src_pts)),3)
    # print points

    X = []
    X_dash = []
    for j in range(3):
        X.append([src_pts[points[j]][0], src_pts[points[j]][1], 0, 0, 1, 0])
        X.append([0, 0, src_pts[points[j]][0], src_pts[points[j]][1], 0, 1])
        X_dash.append(dst_pts[points[j]][0])
        X_dash.append(dst_pts[points[j]][1])

    X = np.array(X)
    X_dash = np.array(X_dash)
    # X_inv = np.linalg.inv(X)
    # A = np.dot(X_inv,X_dash)
    try:
        np.linalg.solve(X, X_dash)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            continue
    A = np.linalg.solve(X, X_dash)
    # print A

    slope = [[A[0], A[1]], [A[2], A[3]]]
    intercept = [A[4], A[5]]

    temp = []
    count = 0
    for l in range(len(src_pts)):
        transformedPoint = np.dot(slope, src_pts[l]) + intercept
        originalPoint = dst_pts[l]
        if distance.euclidean(originalPoint, transformedPoint) < 10:
            count += 1
            temp.append(l)
    # print count

    if count > maxCount:
        maxCount = count
        inliers = temp[:]
        bestSlope = slope[:]
        bestIntercept = intercept[:]

# print maxCount

transformedInliers = []
for i in range(len(inliers)):
    transformedPoint = np.dot(bestSlope, src_pts[inliers[i]]) + bestIntercept
    transformedInliers.extend((transformedPoint[0], transformedPoint[1]))

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

'''

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

h,w = img1.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.CV_AA)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = drawMatches(img1,kp1,img2,kp2,good)

plt.imshow(img3, 'gray'),plt.show()

'''
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

# plt.imshow(img3),plt.show()