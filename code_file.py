import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

x1 = 	np.array([[-5.01,-5.43,1.08,0.86,-2.67,4.94,-2.51,-2.25,5.56,1.03],
				[-8.12,-3.48,-5.52,-3.78,0.63,3.29,2.09,-2.13,2.86,-3.33],
				[-3.68,-3.54,1.66,-4.11,7.39,2.08,-2.59,-6.94,-2.26,4.33]])

x2 = 	np.array([[-0.91,1.30,-7.75,-5.47,6.14,3.60,5.37,7.18,-7.39,-7.50],
				[-0.18,-2.06,-4.54,0.50,5.72,1.26,-4.63,1.46,1.17,-6.32],
				[-0.05,-3.53,-0.95,3.92,-4.85,4.36,-3.65,-6.66,6.30,-0.31]])

x3 = 	np.array([[5.35,5.12,-1.34,4.48,7.11,7.17,5.75,0.77,0.90,3.52],
		[2.26,3.22,-5.31,3.42,2.39,4.33,3.97,0.27,-0.43,-0.36],
		[8.13,-2.66,-9.87,5.19,9.21,-0.98,6.65,2.41,-8.71,6.43]])

miu1 =	np.mean(x1, axis=1)
miu2 =	np.mean(x2, axis=1)
miu3 =	np.mean(x3, axis=1)

cov1 =	np.cov(x1)
cov2 =	np.cov(x2)
cov3 =	np.cov(x3)


def mahalan_dist(x, miu, cov):
	d = np.dot(np.dot((x - miu).T, np.linalg.inv(cov)), (x - miu))
	return d
def g(x, miu, cov, prior=1):
	#  - (0.5*len(cov)*np.log(np.pi/2.0)) removed as constant for all classes
	g_i = (-0.5 * np.dot(np.dot((x - miu).T, np.linalg.inv(cov)), (x - miu))) - (0.5*np.log(np.linalg.det(cov))) + np.log(prior)
	return g_i

def MAP(x):
	g1 = g(x, miu1, cov1, 0.8)
	g2 = g(x, miu2, cov2, 0.1)
	g3 = g(x, miu3, cov3, 0.1)
	c = 1
	m = g1
	if g2 > m:
		m = g2
		c = 2
	if g3 > m:
		c = 3
	return c 


def ML(x):
	g1 = g(x, miu1, cov1)
	g2 = g(x, miu2, cov2)
	g3 = g(x, miu3, cov3)
	c = 1
	m = g1
	if g2 > m:
		m = g2
		c = 2
	if g3 > m:
		c = 3
	return c

print "Classification: Using MAP"
print "P1 (1, 2, 1) is in class: ", MAP(np.array([1, 2, 1]))
print "P2 (5, 3, 1) is in class: ", MAP(np.array([5, 3, 1]))
print "P3 (0, 0, 0) is in class: ", MAP(np.array([0, 0, 0]))
print "P4 (1, 0, 0) is in class: ", MAP(np.array([1, 0, 0]))

print "Classification: Using ML"
print "P1 (1, 2, 1) is in class: ", ML(np.array([1, 2, 1]))
print "P2 (5, 3, 1) is in class: ", ML(np.array([5, 3, 1]))
print "P3 (0, 0, 0) is in class: ", ML(np.array([0, 0, 0]))
print "P4 (1, 0, 0) is in class: ", ML(np.array([1, 0, 0]))
#Plotting MAP

fig1 = pylab.figure()
aax = Axes3D(fig1)

aax.set_xlabel("x1 - feature axis")
aax.set_ylabel("x2 - feature axis")
aax.set_zlabel("x3 - feature axis")
aax.set_title("MAP")
c1x = []
c1y = []
c1z = []
c2x = []
c2y = []
c2z = []
c3x = []
c3y = []
c3z = []
for f1 in np.arange(-10,10, 0.5):
	for f2 in np.arange(-10,10, 0.5):
		for f3 in np.arange(-10,10, 0.5):
			cl = MAP(np.array([f1, f2, f3]))
			if cl == 1:
				c1x.append(f1)
				c1y.append(f2)
				c1z.append(f3)
			if cl == 2:
				c2x.append(f1)
				c2y.append(f2)
				c2z.append(f3)
			if cl == 3:
				c3x.append(f1)
				c3y.append(f2)
				c3z.append(f3)
aax.scatter(c1x, c1y, c1z, c='r', marker='x')
aax.scatter(c2x, c2y, c2z, c='g', marker='o')
aax.scatter(c3x, c3y, c3z, c='y', marker='p')
#pyplot.show()


#Plotting ML
fig = pylab.figure()
ax = Axes3D(fig)

ax.set_xlabel("x1 - feature axis")
ax.set_ylabel("x2 - feature axis")
ax.set_zlabel("x3 - feature axis")
ax.set_title("ML")
c1x = []
c1y = []
c1z = []
c2x = []
c2y = []
c2z = []
c3x = []
c3y = []
c3z = []
for f1 in np.arange(-10,10, 0.5):
	for f2 in np.arange(-10,10, 0.5):
		for f3 in np.arange(-10,10, 0.5):
			cl = ML(np.array([f1, f2, f3]))
			if cl == 1:
				c1x.append(f1)
				c1y.append(f2)
				c1z.append(f3)
			if cl == 2:
				c2x.append(f1)
				c2y.append(f2)
				c2z.append(f3)
			if cl == 3:
				c3x.append(f1)
				c3y.append(f2)
				c3z.append(f3)
ax.scatter(c1x, c1y, c1z, c='r', marker='x')
ax.scatter(c2x, c2y, c2z, c='g', marker='o')
ax.scatter(c3x, c3y, c3z, c='y', marker='p')
pyplot.show()
