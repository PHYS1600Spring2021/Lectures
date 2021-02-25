#!/usr/bin/env python
import numpy as np


def quadratic(x, a):
    return a*x**2

parabola = lambda x, a: a*x**2

# Fibonacci numbers module
def fib1(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while b < n:
        print(b),
        a, b = b, a+b

def fib2(n): # return Fibonacci series up to n
    result = []
    a, b = 0, 1
    while b < n:
        result.append(b)
        a, b = b, a+b
    return result

def fib3():
    # return the elements of a Fibonacci seris using generators
    # generators are objects you can loop over like a list, but
    # the contents are not stored in memory -> usefull in cases
    # where it is cumbersome, or not possible, to store full
    # sequence in memory
    # also do not need to wait for all elements to be generated
    # so there is performance imrovements.
    #
    a, b  = 0, 1
    while True:
        yield a
        a, b = b, a + b


print('*****   Hello PHYS 1600 !!!   *****\n')
#Use the point by point method
fib1(10)

#Use the list append method
print(fib2(10))

# Use the generator
f = fib3()
for i in range(5):
    print(i ,',',next(f))
