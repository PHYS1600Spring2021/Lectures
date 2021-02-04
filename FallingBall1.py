#! /usr/bin/env python
# initial conditions
y0 = 10.0
v0 = 0.0

# keep track of time in this variable
t = 0.0

# timestep
dt = 0.001

y = y0
v = v0
g = 9.8

for i in range(1000):
	y = y + v * dt
	v = v - g * dt
	t = t + dt
	
print('\n' + 15*'**'+' Results ' + 16*'**' +  '\n')
print('final time  = ', t)
print('y = ', y, 'and v = ', v)
print('EXACT y = ', y0 + v0 * t - 0.5 * g * pow(t, 2.0))
print(35*'**')
