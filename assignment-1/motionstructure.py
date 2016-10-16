import scipy.io as sio
import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

dict_points = sio.loadmat("sfm_points.mat")
image_points = dict_points['image_points']
total = image_points.shape[1]
frames = image_points.shape[2]

x_sum = np.sum(image_points[0], axis=0)
y_sum = np.sum(image_points[1], axis=0)

# print x_sum
# print y_sum

t_i = []
for i in range(frames):
    t_i.append([x_sum[i]/total, y_sum[i]/total])

print t_i[0]

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
W = []
for i in range(frames):
    W.append(image_points_x[i])
    W.append(image_points_y[i])

W = np.matrix(W)
U, D, V = np.linalg.svd(W)
M = [np.array(U.T[:3][i]).ravel() * D[i] for i in range(3)]
M = np.matrix(M).T
print np.matrix([np.array(M[0]).ravel(), np.array(M[1]).ravel()])

S = V[:3]

# print np.array(S[0]).ravel()[:10]
# print np.array(S[1]).ravel()[:10]
# print np.array(S[2]).ravel()[:10]
print zip(np.array(S[0]).ravel()[:10], np.array(S[1]).ravel()[:10], np.array(S[2]).ravel()[:10])

fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(np.array(S[0]).ravel(), np.array(S[1]).ravel(), np.array(S[2]).ravel())
# pyplot.show()
pyplot.savefig("box.jpg")