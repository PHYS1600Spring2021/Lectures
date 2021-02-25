import numpy as np


class shapes():

    def __init__(self):
        self.shape = 'shape'
        print("Init'd super class")

    def __str__(self):
        return "This is a {}.".format(self.shape)



class regular_polygon(shape):

    def __init__(self,n_sides, edge_length):

        super().__init__()

        self.shape = 'regular polygon'
        self.edge_length = edge_length
        self.n_sides = n_sides
        self.int_angle = (n_sides-2)*180/n_sides

    def __str__(self):
        out1 = super().__str__()
        return out1 + "with {} sides".format(self.n_sides)


    # special methods to change how the addition and multiplication operators work
    # there are other special methods as well

    def __add__(self,s):
        new_edge = self.edge_length + s

        return regular_polygon(self.n_sides, new_edge)

    def __mul__(self,s):
        scaled_edge = self.edge_length * s

        return regular_polygon(self.n_sides,scaled_edge)

    # some operations may not commute, so we need to explicitly define
    #  a*b = b*a and a+b = b+a
    __radd__ = __add__
    __rmul__ = __mul__

    def perimeter(self):
        return self.edge_length*self.n_sides

    def area(self):
        A = 0.25*self.n_sides*self.edge_length**2/np.tan(np.pi/self.n_sides)

        return A



class square(regular_polygon):

    def __init__(self,edge_length):

        super().__init__(4, edge_length)

        self.shape = 'square'


    #def __mul__(self,s):

        #scaled_edge = self.edge_length * s
        # self.edge_length = scaled_edge

        #return square(scaled_edge)
