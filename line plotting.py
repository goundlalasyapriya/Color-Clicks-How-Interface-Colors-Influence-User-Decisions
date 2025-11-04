import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4])
y=x*2
plt.plot(x,y)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Line Plotting")
plt.show()
plt.figure()
x1=[2,4,6,8]
y1=[3,5,7,9]
plt.plot(x1, y1, 'o:')
plt.show()