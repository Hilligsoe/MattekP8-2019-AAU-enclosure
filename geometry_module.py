# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:02:32 2019
Evt. pakker statesmodel and pysal

@author: cht15
"""

import numpy as np
import matplotlib.pyplot as plt


class spatial_geometry:
    """
    Class of spatial geometries and tools to plot or create them
    """
    def __init__(self, geometry="box", **kwargs):
        self.geometry = geometry

        if 'points' in kwargs:
            self._point_geometry(kwargs['points'])
        elif geometry is "box" and 'length' in kwargs:
            self._box_geometry(kwargs['length'], unit=kwargs.get('unit'))

    def _box_geometry(self, a, unit="m"):
        """
        Creates a box geometry for the room using wall length.
        #TODO Check theird dimention

        Input:
            # a (array), The length of the walls of the rectangular room
            # unit (str), Purely conventional if need for measuring unit arrise

        Output:
            The points defining the polygon shape of the room (always convex)
            is returned to the class object.
        """
        p = np.array(a, dtype=np.float32)

        if p.shape != (2,) and p.shape != (3,):
            raise ValueError("p must be a vector of length 2 or 3.")

        self.dim = p.shape[0]
#        self.wall_names = ['west', 'north', 'east', 'south']
#        if self.dim == 3:
#            self.wall_names += ['floor', 'ceiling']

#        for i in range(self.dim*2):
        self.points = np.array([[0, 0], [p[0], 0], [p[0], p[1]], [0, p[1]]])
        self._point_geometry(self.points)

        if self.dim is 3:  # The Theird dimension is created using vstack
            # The floor is the bonderies of the room along with hight of floor
            self._floor = np.vstack((self.points.T, np.repeat(0, 4))).T
            # The ceiling is the bonderies of the room along with the hight
            self._ceil = np.vstack((self.points.T, np.repeat(p[2], 4))).T

    def _circle_geometry(self, r=1, N=100):
        """
        Creates N points on the circumference of a circle with radius r
        """
        self.points = np.array([[np.cos(2*np.pi/N*i)*r,
                                 np.sin(2*np.pi/N*i)*r] for i in range(N)])

    def _function_geometry(self, f, N, support, dim=2, *args):
        """
        Give point values from a function needed to describe a room along with
        the amount of points needed to create a satisfying resolution of the
        room.
        #TODO Implement it
        #TODO Implement 3 dimensional functions
        """

        try:
            t = np.linspace(support[0], support[1], N)
        except TypeError:
            t = np.linspace(0, support, N)
        except TypeError:
            raise TypeError("The support for the function needs to be a\
                            values / vector")

        self.points = f(t, args)
        raise NotImplementedError('As of 0.01 this feature is not implemented')

    def _point_geometry(self, points):
        """
        Groups points in a list together to form all the points for a wall.
        """
        self._wall = {}
        self._n_wall = len(points)
        self._index = np.arange(self._n_wall+1) % self._n_wall
        for i in range(self._n_wall):
            self._wall[i] = (points[i], points[(i+1) % self._n_wall])

    def _plot_2D_wall(self):
        plt.plot(self.points.T[0][self._index],
                 self.points.T[1][self._index], color='C0')

    def _array_points(self, p):
        """
        No idear what the points were of this (get it?)
        """
        raise NotImplementedError
        p = np.array(p, dtype=np.float32)

        if p.shape != (2,) and p.shape != (3,):
            raise ValueError("p must be a vector of length 2 or 3.")
        self.points = p
