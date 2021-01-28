import sys
import os
import numpy
import matplotlib.pyplot as plt

print('*****   Hello PHYS 1600 !!!   *****\n')

for p in sys.path:
    print(p)

x = numpy.linspace(0,6,100)
plt.plot(x, numpy.sin(x*numpy.pi))
plt.ylabel('sin(x) things')
plt.xlabel('x (1/$\pi$)')
plt.show()

