# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:28:36 2022

@author: harish p jain

This file contains functions to load different properties 
into arrays or other data structures
"""

import numpy as np
import csv
import glob
import re
import pandas as pd

def load_pos(input_dir):
    """
    Parameters
    ----------
    input_dir : string
        The directory which contains the position directory with neopositions
        csv files

    Returns
    -------
    positions_raw : list of panda databases
        Each database corresponds to a given cell. It stores information for 
        every *th timestep about different cell properties like position of 
        center of mass, velocity of center of mass, ..
    
    ranks : int array
        An array of ranks or cell indices
    """
    ranks = []
    positions_raw = []
    file_pattern = input_dir + '/positions/neo_positions_p*.csv'
    for filename in glob.glob(file_pattern):
        tmp = pd.read_csv(filename)
        positions_raw.append(tmp[['time','rank','x0','x1','r','S0','S1','v0','v1', 
                                  'total_interaction', 'neighbours', 
                                  'confine_interaction', 'growth_rate', 
                                  'S0full', 'S1full']])
        # we also need to extract the rank to build the bridge to vtk files
        ranks.append(int(re.findall(r'\d+', filename)[-1]))
    ranks = np.array(ranks)
    
    sorted_index = np.argsort(ranks)
    positions_raw.sort(key= lambda elem: elem.iloc[0]['rank'])
    ranks = ranks[sorted_index]
    
    return positions_raw, ranks

def load_grid(input_dir):
    """
    Parameters
    ----------
    input_dir : string
        The directory with the phasefield directory that contains numpy grid 
        files

    Returns
    -------
    grid_X : 2D numpy float array
        Grid of x positions
    grid_Y : 2D numpy float array
        Grid of y positions

    """
    grid_X = np.load(input_dir + '/phasefield/grid_x.npy')
    grid_Y = np.load(input_dir + '/phasefield/grid_y.npy')
    
    return grid_X, grid_Y

def load_rankfield(input_dir):
    """
    Parameters
    ----------
    input_dir : string
        The directory with the phasefield directory that contains numpy 
        rankfield files

    Returns
    -------
    phasefield : 3D numpy array
        The first dimension corresponds to the time. The next two indices 
        store the rank field for that time. Such that the field is equal to 
        rank of the cell occupying that grid point
    times : float array
        array of time where the phasefield is stored at

    """
    times = np.load(input_dir + '/phasefield/timesteps.npy')
    sizes = load_grid(input_dir)[0].shape
    rankfield = np.zeros([len(times), sizes[0], sizes[1]])
    
    for indt, time in enumerate(times):
        rankfield[indt] = np.load(input_dir + '/phasefield/phi_field' +
                             '{:06.3f}'.format(time) +'.npy')
    return rankfield, times

def load_fields(input_dir, prop='velocity'):
    """

    Parameters
    ----------
    input_dir : string
        input directory
    prop : string, optional
        This is the name of the property for which you would like the field of.
        The possible options are 
        1. 'velocity'
        2. 'normalised nematic' : The two independeny shape tensor components 
            are normalised such that the magnitude of S is 1
        3. 'nematic' : Returns the two independent shape tensor components
        4. 'velocity angle' : Returns the velocity angle field in [pi, pi]
        5. 'nematic angle' : Returns the nematic orientation field in 
            [pi/2, pi/2]
        The default is 'velocity'.


    Returns
    -------
    one or more 3D array
        First axis is time and the next two are properties at the grid point
        corresponding to the cell that occupies that grid point

    """
    positions, ranks = load_pos(input_dir)
    rankfield, times = load_rankfield(input_dir)

    def get_field(a):
        """
        Parameters
        ----------
        a : 2D array
            First index corresponds to time and second to rank. Stores a property

        Returns
        -------
        a_field : 3D array
            First index is time and the next two indices marks each cell by 
            its property

        """
        a_field = np.zeros(rankfield.shape)
        for indt, time in enumerate(times):
            for indr, rank in enumerate(ranks):
                a_field[indt][rankfield[indt]==(rank)] = a[indt, indr]
        return a_field
    
    #2D properties
    prop2D = ['velocity', 'normalised nematic', 'nematic']
    if (prop in prop2D):
        prop0, prop1 = load_property(input_dir, prop)
        return get_field(prop0), get_field(prop1)
    #1D properies
    prop1D = ['velocity angle', 'nematic angle']
    if prop in prop1D:
        return get_field(load_property(input_dir, prop))
    
def load_property(input_dir, prop='velocity'):
    """

    Parameters
    ----------
    input_dir : string
        input directory
    prop : string, optional
        This is the name of the property for which you would like the field of.
        The possible options are 
        1. 'velocity'
        2. 'normalised nematic' : The two independeny shape tensor components 
            are normalised such that the magnitude of S is 1
        3. 'nematic' : Returns the two independent shape tensor components
        4. 'velocity angle' : Returns the velocity angle field in [pi, pi]
        5. 'nematic angle' : Returns the nematic orientation field in 
            [pi/2, pi/2]
        The default is 'velocity'.


    Returns
    -------
    one or more 2D array
        First axis is time and the second axis is the property of the cell and 
        is indexed by rank

    """
    positions, ranks = load_pos(input_dir)
    rankfield, times = load_rankfield(input_dir)

    if prop == 'velocity':
        v0 = np.zeros([len(times), len(ranks)])
        v1 = np.zeros([len(times), len(ranks)])
        for rank in ranks:
            v0[:, rank] = positions[rank]['v0']
            v1[:, rank] = positions[rank]['v1']
        return v0, v1
    
    if prop == 'normalised nematic':
        S0 = np.zeros([len(times), len(ranks)])
        S1 = np.zeros([len(times), len(ranks)])
        for rank in ranks:
            S0[:, rank] = positions[rank]['S0']
            S1[:, rank] = positions[rank]['S1']
        return S0, S1
        
    if prop == 'nematic':
        S0 = np.zeros([len(times), len(ranks)])
        S1 = np.zeros([len(times), len(ranks)])
        for rank in ranks:
            S0[:, rank] = positions[rank]['S0full']
            S1[:, rank] = positions[rank]['S1full']
        return S0, S1
    
    if prop == 'velocity angle':
        v0 = np.zeros([len(times), len(ranks)])
        v1 = np.zeros([len(times), len(ranks)])
        for rank in ranks:
            v0[:, rank] = positions[rank]['v0']
            v1[:, rank] = positions[rank]['v1']
        vel_orient = np.arctan2(v1, v0)
        return vel_orient
    
    if prop == 'nematic angle':
        S0 = np.zeros([len(times), len(ranks)])
        S1 = np.zeros([len(times), len(ranks)])
        for rank in ranks:
            S0[:, rank] = positions[rank]['S0full']
            S1[:, rank] = positions[rank]['S1full']
        nem_orient = np.multiply(np.sign(S1),
                                 (np.arctan(S0/np.abs(S1))
                                  /2.0 + np.pi/4.0))
        return nem_orient