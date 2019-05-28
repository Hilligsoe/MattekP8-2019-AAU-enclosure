# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:02:44 2019

@author: cht15
"""

import time
import numpy as np
from Gibbs_optim import Sampling

np.random.seed(1235)
# =============================================================================
# Init parametres (Standard)
# =============================================================================

# Room parameters

length = (7, 7, 2.4)  # Dim of the room, (lenght, width, height)
s_location = ((70, 70, 24), (139, 120, 36))  # Speaker location in the lattice
modes = True
table = False  # {'x,y,z': (40, 0, 35), 'p,q,r': (20, 30, 0)}  # Optional table

# =============================================================================
# Init parametres for opimisation (Note min_length needs to be 2)
# =============================================================================
# Optimization parameters
iterations = 2000
min_length = 2
T_start = 10
# beta = np.exp(-np.arange(iterations) * 0.005) * T_start + 1
beta = 1 - np.exp(-np.linspace(0, 10, iterations))
annealing = beta

# alternative beta function
lin_iter = 1500
_temp = np.linspace(0, 1, lin_iter)
test_beta = np.hstack((_temp, np.ones(iterations-lin_iter)))
# annealing = test_beta


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

N = 100
check_save = np.zeros(N, dtype=bool)
time_save = np.zeros(N)
for i in range(N):
    start_timer = time.time()
    room = Sampling(length, s_location=s_location, modes=modes, n_mics=6,
                    i_range=i_range, table=table)
    check, s_check = room.Sampler(iterations,
                                  min_length,
                                  v=True,
                                  annealing=annealing,
                                  i_range=i_range)
    check_save[i] = np.all(check == 0) and np.all(s_check)
    stop_timer = time.time()
    time_save[i] = stop_timer - start_timer
#    print(f'It took {stop_timer-start_timer} seconds to run the sampler')

print(f'Average number of completions {np.count_nonzero(check_save)/N}, in\
 percentage {(100/N)*np.count_nonzero(check_save)} ')
print(f'Average time for each rerun {np.mean(time_save)}')
