# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:19:38 2019

@author: cht15
"""

import numpy as np


class objects:
    def table(self, room_array, table, restricted=1, res=0.05):
        """
        Creates and places a recrited area in a room array, from the
        coordinates and dimensions of a table.

        Parameters
        ----------
        room_array : array_like
            The room array for placement of table
        table : dict
            The dict with both the coorner location of a table (x,y,z) as well
            as the dimensions of the table (p,q,r), the dict should be called
            table = {'x,y,z': (x,y,z), 'p,q,r': (p,q,r)}
        restricted : float
            The range for which the room should be restricted from a reflective
            surface.
        res : float
            The distance (in meters) between each point in the room_array

        Returns
        -------
        room_array : array_like
            The updated room_array.
        """
        _restrict = int(restricted/res)
        _min = np.zeros(3, dtype=int)
        _max = np.zeros(3, dtype=int)
        _min[0] = table['x,y,z'][0] - _restrict
        _min[1] = table['x,y,z'][1] - _restrict
        _min[2] = table['x,y,z'][2] - _restrict
        _max[0] = table['x,y,z'][0] + table['p,q,r'][0] + 1 + _restrict
        _max[1] = table['x,y,z'][1] + table['p,q,r'][1] + 1 + _restrict
        _max[2] = table['x,y,z'][2] + table['p,q,r'][2] + 1 + _restrict
        for i in range(3):
            if _min[i] < 0:
                _min[i] = 0
            if _max[i] > np.shape(room_array)[i]:
                _max[i] = np.shape(room_array)[i]
        room_array[_min[0]: _max[0],
                   _min[1]: _max[1],
                   _min[2]: _max[2]] = 99  # Lable locations in room.
        return room_array
