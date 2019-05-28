# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 09:42:28 2019

@author: cht15
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry_module import spatial_geometry
from object_module import objects
from modes_module import modes

# =============================================================================
# Class elements
# =============================================================================


class Sampling(spatial_geometry, modes, objects):
    """
    Creates a room and discretise it to final point latice (simplex).
    Then init P microphones in the room by uniformly distribution.

    As the locations of interest are restricted to locations where a microphone
    is located. As such a itteration over each point is performed.
    (For itterative condition mode each microphone is moved to the point
    in its neighbourhood which leads to the maximum)
    (For Gibbs two random points are taking into consideration (0 and 1)
    is only moved if it reduces the energy (not just the max)).

    The room is only indexed by values where a microphone is allowed.
    As such any considerations of walls an objects can be discarded.

    0 is empty, 1 is microphone, 2 is speaker, 3 is modes, 4 is active subject,
    5 is testsubject.
    """

# TODO Fully integrate rooms as points.
# TODO Implement the different function a, b and then make the sweep.
    def __init__(self, length,
                 s_location=None,
                 i_range=1,
                 res=0.05,
                 n_mics=6,
                 Reduced=False,
                 restricted=1,
                 octave=np.array([89, 177]),
                 c=343,
                 table=False,
                 modes=True, **kwargs):
        """
        Given the parameters sets up the room as an array.
            Restriced to a square room (as a start).

        Parameters
        ----------
        length : array_like
            x,y,(z) coordinates of a rooms corners.
        s_location : array_like
            x,y,(z) coordinates for the location of speakers in the room
            (Optional default off)
        i_range : float
            Interaction range of speaker to block out of the room.
        res : float
            Resolution of the room (default: 0.05). This corresponds to a
            spacing between the each point in the array of 5 cm.
        n_mics : int
            The minimum number of microphone needed in the system. (default: 6)
        Reduced : Boolean
            True false parameter changing whether the room measurements (Wall
            lengths and objects) are with or without considering microphone
            restrictions. (default: False)
            (i.e. default is regular room measurements)
        restricted : float
            The minimum length a microphone needs to be placed from a wall.
        octave : array_like
            The max and min frequency for octave range used to calculate the
            nodel planes.
        c : float
            Speed of sound in meters per second
                (default 343 speed of sound at 20C)
        table : dict
            Whether the system should place a box shaped object or not.
            The dict should contain two elements (x,y,z) coordinates of the
            corner closest to (0,0,0)
        modes : boolean
            Whether the system should account for the normal modes of the room.

        Class elements
        --------------
        Room : array_like
            M,N,O matrix which indexes the locations in the room.
                Zero is empty space, Ones are michrophone,
                Twos are speakers and NaN are outside/restricted areas.
                Threes are modes in the room.
        """
        self.res = res
        self.n_mics = n_mics
        self.octave = octave
        self.c = c

        if Reduced:
            try:
                self.length = length + 2*restricted
            except TypeError:
                self.length = np.array([x + 2*restricted for x in length])
        else:
            self.length = np.asarray(length)
        self.room_array = np.zeros(
                            np.array([int(self.length[i]//self.res)+1
                                      for i in range(self.length.shape[0])]))
        self.shape = np.asarray(self.room_array.shape)

        if modes:
            self.room_array = self.normal_modes(self.room_array, self.length,
                                                self.octave, self.res, self.c)
        if s_location is None:
            self.speakers = False
            pass
        else:
            try:
                self.speakers = {}
                len(s_location[0])
                __shape = (len(s_location))
                for i in range(__shape):
                    self.room_array[tuple(s_location[i])] = 2
                    self.speakers[f'room_{i}'] = self._subroom(
                            np.asarray(s_location[i]), init=True,
                            i_range=i_range)
                    self.speakers[f'location_{i}'] = tuple(s_location[i])
                    self.speakers[f's_location_{i}'] = np.argwhere(
                        self.speakers[f'room_{i}'] == 2)
                self.speakers['n'] = __shape
            except TypeError:
                self.speakers = {}
                self.room_array[tuple(s_location)] = 2
                self.speakers[f'room_0'] = self._subroom(
                        np.asarray(s_location), init=True, i_range=i_range)
                self.speakers[f'location_0'] = tuple(s_location)
                self.speakers['n'] = 1
                self.speakers[f's_location_0'] = np.argwhere(
                        self.speakers[f'room_0'] == 2)

        if table:
            self.room_array = self.table(self.room_array, table,
                                         restricted=restricted, res=self.res)
        _rm = int(restricted/res)  # values to be removed from the edge.
        self._plot_save_array = self.room_array
        self._plot_save_length = self.length
        self.room_array = self.room_array[_rm: -_rm, _rm: -_rm].T[_rm: -_rm].T
        self.length = self.length - 2*restricted
        if np.any(self.length < 0):
                raise ValueError('Room to small to satify restrictions')
        self._discretise_room()
        super().__init__(length=self._plot_save_length)

    def _discretise_room(self):
        """
        Discretise a rectangular room and init it with n_mics uniformly placed
        microphones in the avialble locations.
        This should only to be called in the init function.

        Class elements
        --------------
        Room : array_like
            M,N,O matrix which indexes the locations in the room.
                Zero is empty space, Ones are michrophone,
                Twos are speakers and NaN are outside/restricted areas.
        """
        poss_loc = np.nonzero(self.room_array == 0)  # Possible locations
        # Random index
        r_index = np.random.randint(poss_loc[0].shape[0], size=6)
        self.room_array[poss_loc[0][r_index],
                        poss_loc[1][r_index],
                        poss_loc[2][r_index]] = 1

    def _subroom(self, Location, init=False, i_range=1):
        """
        Given a room array and a locations calculate the neighbourhood array.

        Parameters
        ----------
        Location : int
            Center of the neighbourhood of interest.
        init : bool
            If False (First time called for creating microphones)
        i_range : float
            The index range in meters for which microphones interact.
        """
        # Calculate the index range for the subroom

        if init:
            self._index_r = int(i_range/self.res)
        s_i_min = Location - self._index_r
        s_i_min[s_i_min < 0] = 0

        try:
            s_i_max = Location + self._index_r + 1
            max_index = s_i_max > self.shape
            s_i_max[max_index] = self.shape[max_index]
            subroom = self.room_array[s_i_min[0]: s_i_max[0],
                                      s_i_min[1]: s_i_max[1],
                                      s_i_min[2]: s_i_max[2]]
        except IndexError:
            s_i_max = Location + self._index_r + 1
            max_index = s_i_max > self.shape
            s_i_max[max_index] = self.shape[max_index]
            subroom = self.room_array[s_i_min[0]: s_i_max[0],
                                      s_i_min[1]: s_i_max[1]]
        return subroom

    def _H(self, subroom, i_range=2, beta=1, test=False, ):
        """
        Energyfunction for the model

        Parameters
        ----------
        Subroom : array_like
            Hypothical subset of the room.
        i_range : float
            Interaction range of speakers (default 2)
        beta : float
            Scalar value for the energi function (default 1)
        test : boolean
            Whether the system should be looking for active subject or test
            subject. (Slightly reduces computational cost.)

        Returns
        -------
        Energy : float
            The energy of the subroom
        """

        if test is False:
            locations = np.argwhere(subroom == 1)
            s = np.argwhere(subroom == 4)[0]
            H = 0
            _temp_true = 0
            for i in range(locations.shape[0]):
                _temp = (np.linalg.norm(locations[i] - s, ord=2) * self.res)
                H += 1/_temp
                if _temp < 2 and _temp != 0:
                    _temp_true += 1
            clear = True
            if self.speakers:
                _s = np.argwhere(self._plot_save_array == 4)
                for i in range(self.speakers['n']):
                    _temp = (np.linalg.norm(self.speakers[f'location_{i}']-_s,
                                            ord=2) * self.res)
                    if _temp < i_range:
                        H += 1/(_temp*0.5)
                        clear = False
            return np.exp(-beta*H), _temp_true, s, clear
        else:
            locations = np.argwhere(subroom == 1)
            s = np.argwhere(subroom == 5)
            H = 0
            for i in range(locations.shape[0]):
                H += 1/(np.linalg.norm(locations[i] - s, ord=2) * self.res)
            if self.speakers:
                _s = np.argwhere(self._plot_save_array == 5)
                for i in range(self.speakers['n']):
                    _temp = (np.linalg.norm(self.speakers[f'location_{i}']-_s,
                                            ord=2) * self.res)
                    if _temp < i_range:
                        H += 1/(_temp*0.5)
            return np.exp(-beta*H)

    def Sampler(self, max_iter, i_r, annealing=False, v=False,
                i_range=2, **kwargs):
        """
        Gibbs sampling of the Gibbs random field - Iterating over all
        microphone locations picking a neighbour and checking whether the
        microphone should mode to the location by calculating the energy H(X).

        Parameters
        ----------
        self.room_array : array_like
            The room defined as an array for updating and placing microphones.
        max_iter : int
            The maximum number of iterations the Gibbs sampler are allowed to
            perform.
        i_r : float
            Interaction Range - The range for which the neighbourhood is
            defined around S.
        annealing : str/Boolean
            Whether to apply annealing (False) or if str, what type of
            annealing to apply.
        v : boolean
            Verbosity - Turn on for console update. (defualt False)
        """
        self._index_r = int(i_r/self.res)
        check = np.zeros(self.n_mics)

        if hasattr(annealing, '__call__'):
            const = annealing(max_iter, **kwargs)
        elif type(annealing) == np.ndarray:
            const = annealing
        else:
            const = np.ones(max_iter)

        for i in range(max_iter):
            mic_locations = np.argwhere(self.room_array == 1)
            if v is True:
                if i == max_iter - 1:
                    print(f'\rCurrent iterations {i+1} out of {max_iter}')
                else:
                    print(f'\rCurrent iterations {i+1} out of {max_iter}',
                          end='\r')
            for j in range(self.n_mics):
                subroom = self._subroom(mic_locations[j])  # Neighbourhood
                _mic_i = tuple(x for x in mic_locations[j])  # Coordinates

                self.room_array[_mic_i] = 4
                H0, neighbours, s, clear = self._H(subroom,
                                                   beta=const[i],
                                                   i_range=i_range)
                sphere_c = np.argwhere(subroom == 4)
                self.room_array[_mic_i] = 0

                possible_mic_location = np.argwhere(subroom == 0)
                np.random.shuffle(possible_mic_location)
                for x in possible_mic_location:
                    if np.linalg.norm(x-sphere_c, ord=2) * self.res < i_r:
                        _test_mic_x = tuple(y for y in x)
                        break

                subroom[_test_mic_x] = 4
                _temp = self._subroom(np.argwhere(self.room_array == 4)[0])
                H, _neighbours, __s, clear = self._H(_temp,
                                                     beta=const[i],
                                                     i_range=i_range)
                subroom[_test_mic_x] = 0
                sample = np.random.uniform()
                if H > sample:
                    s = x
                    neighbours = _neighbours
                check[j] = neighbours
                subroom[tuple(y for y in s)] = 1
            if self.speakers:
                _s_true = np.zeros(self.speakers['n'], dtype=bool)
                _where_m = np.argwhere(self._plot_save_array == 1)
                for i in range(self.speakers['n']):
                    _t = np.linalg.norm(
                            (_where_m-np.asarray(self.speakers[f'location_{i}']
                                                 )), ord=2, axis=1)
                    if np.all(_t*self.res > i_range):
                        _s_true[i] = True
                if np.all(check == 0) and np.all(_s_true):
                    print('\n')
                    break
            else:
                if np.all(check == 0):
                    print('\n')
                    break
        return check, _s_true

# TODO Create an inherited class with sample function/optimizations
    def ICM(self, i_r, i_range=2, v=False):
        """
        Iterated conditional modes - Local maximum of Markov random field by
        iteratively maximizing the probability of each variable conditioned on
        the rest.
        Calculate the energy of the subroom (given as the the neighbourhood.).
        Given a location of a microphone move it to another unoccupied
        locations in its neighbourhood where the conditional probability is
        maximised.

        Parameters
        ----------
        self.room_array : array_like
            The room defined as an array for updating and placing microphones.
        i_r : float
            Interaction Range - The range for which the neighbourhood is
            defined around S.
            Minimum length for speakers should be (343/89)/2)
        v : boolean
            Verbosity - Turn on for console update. (defualt False)
        """
        self._index_r = int(i_r/self.res)
        mic_locations = np.argwhere(self.room_array == 1)
        for i in range(self.n_mics):
            if v is True:
                if i == self.n_mics - 1:
                    print(f'\rCurrent iterations {i+1} out of {self.n_mics}')
                else:
                    print(f'\rCurrent iterations {i+1} out of {self.n_mics}',
                          end='\r')

            subroom = self._subroom(mic_locations[i])  # Neighbourhood of mic_i
            _mic_i = tuple(x for x in mic_locations[i])  # mic_i coordinates
            self.room_array[_mic_i] = 4  # Update mic_i value to 4. (see top.)
            H0, neighbours, s, clear = self._H(subroom, i_range=i_range)
            if neighbours == 0 and clear:
                self.room_array[_mic_i] = 1
            else:
                possible_mic_location = np.argwhere(subroom == 0)
                np.random.shuffle(possible_mic_location)
                for x in possible_mic_location:
                    _test_mic_x = tuple(y for y in x)
                    subroom[_test_mic_x] = 5
                    _temp = self._subroom(np.argwhere(self.room_array == 5)[0])
                    H = self._H(_temp, i_range=i_range, test=True)
                    if H > H0:
                        H0 = H
                        s = x
                    subroom[_test_mic_x] = 0
                    if H == 1:
                        break
                self.room_array[_mic_i] = 0
                subroom[tuple(y for y in s)] = 1

    def Condition_check(self, i_range=2):
        """
        Checks whether the system fullfills the ISO 3382-2 standard.

        Parameters
        ----------
        self.room_array : array_like
            The room defined as an array for updating and placing microphones.
        i_range : float
            The interaction range for speakers in the room.

        Returns
        -------
        Check : boolean
            Does the system adhere to ISO 3382-2
        Unfulfilled : array_like
            Boolean array indicating the microphones which position does adhere
            to the standard.
        """
        mic_locations = np.argwhere(self.room_array == 1)
        unfulfilled_locations = np.array([True for i in range(self.n_mics)])
        for i in range(self.n_mics):
            subroom = self._subroom(mic_locations[i])
            if np.count_nonzero(subroom == 1) > 1:
                mics = np.argwhere(subroom == 1)
                for i in range(len(mics)):
                    _temp = (np.linalg.norm(mics[i]-mics, ord=2, axis=1)
                             * self.res)
                    _temp_true = True
                    for y in _temp:
                        if y < 2 and y != 0:
                            _temp_true = False
                    unfulfilled_locations[i] = _temp_true
        check = np.all(unfulfilled_locations)

        if self.speakers:
            _s_true = np.zeros(self.speakers['n'], dtype=bool)
            _where_m = np.argwhere(self._plot_save_array == 1)
            for i in range(self.speakers['n']):
                _t = np.linalg.norm(
                        (_where_m-np.asarray(self.speakers[f'location_{i}']
                                             )), ord=2, axis=1)
                if np.all(_t*self.res > i_range):
                    _s_true[i] = True

        return bool(check), unfulfilled_locations, _s_true

    def plot2D(self, color='C2', lab='Initial mic.', circ=True):
        """
        Plot function for the 2D room. Given the microphone location and
        restricted room boundery plots illustrate the boundery and places the
        microphone.

        Parameters
        ----------
        self : class element
            Call able with out other parameters.
        Color : str
            The matplotlib color used for the scatter dots. (Default 'C2')
        """
        self._plot_2D_wall()
        mic = (np.argwhere(self._plot_save_array == 1) + 1) * self.res
        if circ:
            for i in range(len(mic)):
                circle1 = plt.Circle((mic[i, :][0], mic[i, :][1]), 2,
                                     fill=False, color=color)
                plt.gcf().gca().add_artist(circle1)
        plt.scatter(mic[:, 0], mic[:, 1], color=color, label=lab)
        plt.xlabel('[m]')
        plt.ylabel('[m]')
        plt.show()

    def plot3D(self, color='C2', subfig=111, element=1, plot_speak=False):
        """
        Plot function for the 3D room. Given the microphone location and
        restricted room boundery plots illustrate the boundery and places the
        microphone.

        Parameters
        ----------
        self : class element
            Call able with out other parameters.
        color : str
            The matplotlib color used for the scatter dots. (Default 'C2')
        subfig : int
            The matplotlib subfigure parameters (see matplotlib) (Default 111)
        element : int
            Element of interest as specified by the script docstring.
        plot_speak : bool
            Whether or not the system should also plot the speaker position
            (Default False)
        """
        Z = np.vstack((self._floor, self._ceil))

        fig = plt.figure('3D')
        self.ax = fig.add_subplot(subfig, projection='3d')

        _ele = (np.argwhere(self._plot_save_array == element) + 1) * self.res
        # plot mics locations
        self.ax.scatter3D(_ele[:, 0], _ele[:, 1], _ele[:, 2], color=color)
        if plot_speak:
            speak = (np.argwhere(self._plot_save_array == 2) + 1) * self.res
            self.ax.scatter3D(speak[:, 0], speak[:, 1], speak[:, 2],
                              color='C1')

        # list of sides' polygons of figure
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                 [Z[4], Z[5], Z[6], Z[7]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[2], Z[6], Z[5]],
                 [Z[4], Z[7], Z[3], Z[0]]]

        # plot sides
        pc = Poly3DCollection(verts, linewidths=1, edgecolors='C0')
        pc.set_alpha(0.1)
        pc.set_facecolors('C3')
        self.ax.add_collection3d(pc)

        self.ax.set_xlim3d(0, self._plot_save_length[0])
        self.ax.set_ylim3d(0, self._plot_save_length[1])
        self.ax.set_zlim3d(0, self._plot_save_length[2])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.show()
