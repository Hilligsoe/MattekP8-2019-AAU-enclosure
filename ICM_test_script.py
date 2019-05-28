# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:32:38 2019

@author: cht15
"""

import time
import numpy as np
from Gibbs_optim import Sampling

# =============================================================================
# Init parametres (Standard)
# =============================================================================

# Room parameters

length = (7, 5, 2.4)  # Dim of the room, (lenght, width, height)
s_location = ((0, 0, 20), (139, 99, 36))  # Speaker location in the lattice
modes = True
table = False  # {'x,y,z': (40, 0, 35), 'p,q,r': (20, 30, 0)}  # Optional table


# =============================================================================
# Init parametres for opimisation (Note min_length needs to be 2)
# =============================================================================
# Optimization parameters
min_length = 2  # minimum length for speakers (should minimum be (343/89)/2)

# set d_min, distance from speaker to mic
def speaker_dist(room, const=0.3, c=343):
    """
    Calculate the interaction range Dmin for speakers, given room dimentions

    Parameters
    ----------
    room : array_like
        (x,y,z) lengths of a square room
    const : float
        Absorbtion constant used for all reflective surfaces (default 0.3)
    c : float
        The speed of sound in meters/seconds (default 343 or speed of sound at
        20 degrees celcius)
    """
    S = (room[0]*room[2]*2*const)+(room[1]*room[2]*2*const)+(room[0]*room[1]
                                                             * 2*const)
    V = room[0]*room[1]*room[2]
    est_rev = 0.16*(V/S)
    dist = 2*np.sqrt(V/(c*est_rev))
    return dist


i_range = speaker_dist(length)

# =============================================================================
# Perfomer optimisation
# =============================================================================
N = 100  # Number of different rooms to run
Internal_n_iterations = 3  # Number of ICM iterations on the same room
check_save = np.zeros(N, dtype=bool)  # Save array for conditions
time_save = np.zeros(N)  # Save array for time
for i in range(N):  # Forloop for performing ICM on the same loop iteratively
    start_timer = time.time()
    room = Sampling(length, s_location=s_location, modes=modes,
                    i_range=i_range)

    for j in range(Internal_n_iterations):

        room.ICM(min_length, i_range=i_range, v=True)
        check, u_locations, mic_check = room.Condition_check(i_range=i_range)
        if check:
            if np.all(mic_check):
                break
    stop_timer = time.time()
    time_save[i] = stop_timer - start_timer
    check_save[i] = check and np.all(mic_check)

print(f'Average number of completions {np.count_nonzero(check_save)/N}')
print(f'Average time for each rerun {np.mean(time_save)}')
