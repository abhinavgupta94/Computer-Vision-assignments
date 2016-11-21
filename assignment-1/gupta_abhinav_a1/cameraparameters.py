import numpy as np
import scipy.linalg

with open('world.txt') as f:
    x_world = np.array(map(float, f.readline().split()))
    y_world = np.array(map(float, f.readline().split()))
    z_world = np.array(map(float, f.readline().split()))

with open('image.txt') as f:
    x_local = np.array(map(float, f.readline().split()))
    y_local = np.array(map(float, f.readline().split()))
    z_local = np.array(map(float, f.readline().split()))

# w array used for converting non-homogeneous matrix to homogeneous matrix
w = np.ones(len(x_world))
# Construct matrix A(20x12) to solve AP = 0
A = []
for i in range(len(x_world)):
    # first, second, third represent the corresponding entries in matrix A for each row
    first = np.zeros(4)
    second = -w[i] * np.array([x_world[i],y_world[i],z_world[i],w[i]])
    third = y_local[i] * np.array([x_world[i],y_world[i],z_world[i],w[i]])
    F = np.hstack((first, second, third))

    first = w[i] * np.array([x_world[i], y_world[i], z_world[i], w[i]])
    second = np.zeros(4)
    third = -x_local[i] * np.array([x_world[i], y_world[i], z_world[i], w[i]])
    S = np.hstack((first, second, third))

    A.append(F)
    A.append(S)
A = np.matrix(A)

# Singular Value Decomposition of A
U, s, V = np.linalg.svd(A)
# Extract the eigenvector corresponding to the smallest eigenvalue and store the camera parameters in P(3x4)
P = V[-1:].reshape(3,4)
print "Camera matrix P"
print P

# Verify cross product of (x, PX)  = 0
transposed = []
for j in range(len(x_world)):
    X = np.array([x_world[j],y_world[j],z_world[j],w[j]])
    x = np.array([x_local[j],y_local[j],w[j]])
    transposed.append(np.cross(x,np.dot(P,X)))
transposed = np.array(transposed).reshape(-1,3)
# print transposed

# Singular Value Decomposition of P
U, s, V = np.linalg.svd(P)
# Calculate world coordinates of projection center of camera C(3x1)
C = np.array(V[-1:]).ravel()
# Convert homogeneous coordinates back to non-homogeneous coordinates
C = np.array([round(C[i]/C[3]) for i in range(len(C)-1)]).reshape(3,-1)
print "Camera world coordinates - 1st method"
print C

# Alternate method
# RQ decomposition of P to separate intrinsic parameters (q) and transformation parameters (r)
q,r = scipy.linalg.rq(P, mode='economic')

# R(3x3) is Rotation matrix and t(3x1) is translation matrix
R = np.matrix(np.transpose(r)[:3]).T
t = np.matrix(np.transpose(r)[3]).T
# Solve for world coordinates C_t(3x1) for t = -RC_T
C_t = np.linalg.solve(-R,t)
print "Camera world coordinates - 2nd method"
print C_t
