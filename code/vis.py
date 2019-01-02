import matplotlib as mpl
mpl.use('Agg')

import numpy as np  
import matplotlib.pyplot as plt  
t = np.arange(-1, 2, .01)  
s = np.sin(2 * np.pi * t)  
  
plt.plot(t,s)  
# draw a thick red hline at y=0 that spans the xrange  
l = plt.axhline(linewidth=4, color='r')  
plt.axis([-1, 2, -1, 2])  
plt.show()  
plt.close()  
  
# draw a default hline at y=1 that spans the xrange  
plt.plot(t,s)  
l = plt.axhline(y=1, color='b')  
plt.axis([-1, 2, -1, 2])  
plt.show()  
plt.close()  
  
# draw a thick blue vline at x=0 that spans the upper quadrant of the yrange  
plt.plot(t,s)  
l = plt.axvline(x=0, ymin=0, linewidth=4, color='b')  
plt.axis([-1, 2, -1, 2])  
plt.show()  
plt.close()  
  
# draw a default hline at y=.5 that spans the the middle half of the axes  
plt.plot(t,s)  
l = plt.axhline(y=.5, xmin=0.25, xmax=0.75)  
plt.axis([-1, 2, -1, 2])  
plt.show()  
plt.close()  
  
plt.plot(t,s)  
p = plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)  
p = plt.axvspan(1.25, 1.55, facecolor='g', alpha=0.5)  
plt.axis([-1, 2, -1, 2])  
plt.savefig('a.png')  
plt.show()  

