# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:07:34 2019

@author: cht15
"""

import numpy as np


def I_check(x, octave):
    return x > octave[0] and x < octave[1]


class modes:
    def normal_modes(self, room_array, length, octave, res, c=343):
        """
        Calculates the locations of the nodel planes for the normal modes, and
        restrict the room_array indexes, such that no microphone can be places
        in a nodel plane.

        Parameters
        ----------
        room_array : array_like
            The room defined as an array for updating and placing microphones.
        length : array_like
            x, y, z dimentions of the room
        octave : array_like
            The max and min frequency for octave range used to calculate the
            nodel planes.
        res : float
            Spacing between each site of the room_array in meters
        c : float
            Speed of sound in meters per second
                (default 343 speed of sound at 20C)

        returns
        -------
        room_array : array_like
            The updated room_array now with the planes corresponding to normal
            modes labled with (3).
        """
        order = 12
        _order = np.arange(order) + 1
        cm_res = res * 100

        # freq_L is an array of all frequencies with normal modes of any order,
        # in the length direction (X axis)
        freq_L = (c*_order)/(2*(length[0]))
        freq_W = (c*_order)/(2*(length[1]))
        freq_H = (c*_order)/(2*(length[2]))

        # orders_L is an array with the orders of the modes corresponding to
        # frequencies within the octave.
        orders_L = np.array(
                [_order[a] for a in range(order) if I_check(freq_L[a], octave)]
                )
        orders_W = np.array(
                [_order[a] for a in range(order) if I_check(freq_W[a], octave)]
                )
        orders_H = np.array(
                [_order[a] for a in range(order) if I_check(freq_H[a], octave)]
                )
        # X axis
        # trans_index = 0
        for i in orders_L:
            # each nodal plane is an odd intergater of the lenght/2*n
            # first list all odd int corresponding to order i.
            odd_temp = np.array([a for a in _order[::2] if a < 2*i])

            # find index for nodals planes and set equal to 3.
            for j in odd_temp:
                index = int(((length[0]*100)/(2*i))*j)  # index in orig. room

                # when index is between to nodes in lattice, round op or down.
                if np.mod(index, cm_res) > cm_res/2:
                    trans_index = int(index/cm_res)+1
                else:
                    trans_index = int(index/cm_res)

                room_array[trans_index] = 3

        # Y axis
        for i in orders_W:
            odd_temp = np.array([a for a in _order[::2] if a < 2*i])
            for j in odd_temp:
                index = int((length[1]*100/(2*i))*j)
                temp = room_array.swapaxes(0, 1)
                # temp room array where Y axis is the first index

                if np.mod(index, cm_res) > cm_res/2:
                    trans_index = int(index/cm_res)+1
                else:
                    trans_index = int(index/cm_res)
                temp[trans_index] = 3

        # Z axis
        for i in orders_H:
            odd_temp = np.array([a for a in _order[::2] if a < 2*i])
            for j in odd_temp:
                index = int((length[2]*100/(2*i))*j)
                temp = room_array.swapaxes(0, 2)
                if np.mod(index, cm_res) > cm_res/2:
                    trans_index = int(index/cm_res)+1
                else:
                    trans_index = int(index/cm_res)
                temp[trans_index] = 3

        return room_array


if __name__ == "__main__":
    res = 0.05
    length = np.array([6, 4, 3])
    octave = np.array([89, 177])
    room_array = np.zeros(
            np.array([int(length[i]//res)+1 for i in range(length.shape[0])]))

    test = modes()
    room = test.normal_modes(room_array, length, octave, res)
