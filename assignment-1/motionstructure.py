import scipy.io as sio
import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

dict_points = sio.loadmat("sfm_points.mat")
image_points = dict_points['image_points']
total = image_points.shape[1]
frames = image_points.shape[2]

# Sum all the x, y coordinates of image points for finding centroid
x_sum = np.sum(image_points[0], axis=0)
y_sum = np.sum(image_points[1], axis=0)

# print x_sum
# print y_sum

# t_i has the translation vectors of all frames
t_i = []
for i in range(frames):
    t_i.append([x_sum[i]/total, y_sum[i]/total])

print "Translation vector of first camera"
print t_i[0]

# Subtact each point with the centroid coordinates
for l in range(frames):
    for i in range(total):
        image_points[0][i][l] = image_points[0][i][l] - t_i[l][0]
        image_points[1][i][l] = image_points[1][i][l] - t_i[l][1]

x_sum = np.sum(image_points[0], axis=0)
y_sum = np.sum(image_points[1], axis=0)

# print x_sum
# print y_sum

image_points_x = np.transpose(image_points[0])
image_points_y = np.transpose(image_points[1])
# Create W(2*frames x total) matrix
W = []
for i in range(frames):
    W.append(image_points_x[i])
    W.append(image_points_y[i])

W = np.matrix(W)
# Singular Value Decomposition of W
U, D, V = np.linalg.svd(W)

# Calculate M(2*frames x 3) matrix by multiplying the first 3 columns of U with the first 3 eigenvalues
M = [np.array(U.T[:3][i]).ravel() * D[i] for i in range(3)]
M = np.matrix(M).T

print "Affine camera matrix for first camera"
print np.matrix([np.array(M[0]).ravel(), np.array(M[1]).ravel()])

# Extract first 3 columns of V for calculating 3D world coordinates
S = V[:3]

print "3D coordinates(x,y,z) for first 10 world points"
print np.array(S[0]).ravel()[:10]
print np.array(S[1]).ravel()[:10]
print np.array(S[2]).ravel()[:10]
# print zip(np.array(S[0]).ravel()[:10], np.array(S[1]).ravel()[:10], np.array(S[2]).ravel()[:10])

# Plotting 3D world points
fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(np.array(S[0]).ravel(), np.array(S[1]).ravel(), np.array(S[2]).ravel())
pyplot.savefig("worldobject.jpg")
# pyplot.show()
