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

w = np.ones(len(x_world))
A = []
for i in range(len(x_world)):
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

U, s, V = np.linalg.svd(A)
P = V[-1:].reshape(3,4)

transposed = []
for j in range(len(x_world)):
    X = np.array([x_world[j],y_world[j],z_world[j],w[j]])
    x = np.array([x_local[j],y_local[j],w[j]])
    transposed.append(np.cross(x,np.dot(P,X)))
transposed = np.array(transposed).reshape(-1,3)
# print transposed

U, s, V = np.linalg.svd(P)
C = np.array(V[-1:]).ravel()
C = [C[i]/C[3] for i in range(len(C)-1)]
print C

q,r = scipy.linalg.rq(P, mode='economic')

R = np.matrix(-1 * np.transpose(r)[:3]).T
t = np.matrix(np.transpose(r)[3]).T
C_t = np.linalg.solve(R,t)
print C_t

