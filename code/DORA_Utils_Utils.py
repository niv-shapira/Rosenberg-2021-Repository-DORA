# Imports
import pickle
import numpy as np
from itertools import chain
from copy import deepcopy

# Markus's code
from MM_Maze_Utils import *
from MM_Traj_Utils import *

#####################
###### GLOBALS ######
#####################

UnrewNames=['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
RewNames=['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
UnrewNamesSub=['B5','B6','B7','D3','D4','D5','D7','D8','D9'] # excluding D6 which barely entered the maze
AllNames=RewNames+UnrewNames

NUM_OF_NODES = 126  # Total number of nodes in the maze
TERMINALS = np.arange(63,127)  # Array of terminal node indices
TERMINAL_MAT = [1,0,0]  # Probability distribution for terminal states
HOME_MAT = [0,1/2,1/2]  # Probability distribution for home state
AZERO = 0.000001  # Small constant to avoid division by zero

#####################
# UTILITY FUNCITONS #
#####################

def NewTransMatrix(ma):
    '''
    Add node 127 to trans matrix
    
    Parameters:
    ma: Maze object
    
    Returns:
    np.array: Updated transition matrix with node 127 added
    '''
    tra = TransMatrix(ma)
    tra = np.append(tra,[[-1,0,0]],axis=0)
    return tra

def ConvertIndicesToParameters(indices, parameters):
    '''
    Convert indices to corresponding parameter values
    
    Parameters:
    indices: List of indices
    parameters: List of parameter lists
    
    Returns:
    list: List of parameter values corresponding to the given indices
    '''
    params = []
    for j,index in enumerate(indices):
        params += [round(parameters[j][indices[j]],1)]
    return params

def EvalStr(fst,f):
    '''
    Evaluate a string with format specifiers
    
    Parameters:
    fst: Format string
    f: Value to be formatted
    
    Returns:
    str: Formatted string
    '''
    return fst % f

def TranslateAction(action):
    '''
    Translate action from original action code to our action code
    
    Parameters:
    action: Original action code
    
    Returns:
    int: Translated action code
    '''
    if (action == -1):
        return -1
    else:
        return (action + 1) % 3

def GetActionFromStates(state0, state1, ma):
    '''
    Get the action that leads from state0 to state1
    
    Parameters:
    state0: Initial state
    state1: Next state
    ma: Maze object
    
    Returns:
    int: Translated action code
    '''
    return TranslateAction(StepType2(state0, state1, ma))  # step type from MM_Maze_Utils

def GetValue(E_state, action):
    '''
    Return E(s,a)
    
    Parameters:
    E_state: Array of state values
    action: Action index (0 - parent, 1 - left, 2 - right)
    
    Returns:
    float: Value of the state-action pair
    '''
    return E_state[action]

def GetColorSize(params):
    '''
    Calculate the total number of parameter combinations
    
    Parameters:
    params: List of parameter lists
    
    Returns:
    int: Total number of parameter combinations
    '''
    size = 1
    for prow in params:
        size = size * len(prow)

    return size

def GetAvgAndSD(l):
    return round(np.mean(l),2), round(np.std(l),2)

def NewNodes(da):
    '''
    Calculate number of new nodes visited in window
    
    Parameters:
    hist_states: Matrix of history
    
    Returns:
    list: [empty array, window sizes, new nodes counts]
    '''
    newnodes = []
    sumofnewnodes = []
    sumofsteps = list(range(len(da)))
    for s in da:
        if s not in newnodes:
            newnodes += [s]
        sumofnewnodes += [len(newnodes)]
    return sumofsteps, sumofnewnodes

def NewEndNodes(da):
    en=list(range(2**6-1,2**(6+1)-1)) # list of node numbers in level le
    ce=da
    ei=np.where(np.isin(ce,en))[0] # index of all the desired node states
    if len(ei)>0: # if there is at least one state
        cn=np.copy(ce[ei]) # only the desired nodes
        lc=len(cn) # number of desired nodes encountered
        c=np.array([2,3,6,10,18,32,56,100,180,320,560,1000,1800,3200,5600,10000]) # window width in nodes
        c=c[np.where(c<lc)] # use only those shorter than full length
        c=np.append(c,lc) # add full length as last value
        n=[np.average(np.array([len(set(cn[j:j+c1])) for j in range(0,lc-c1+1,(lc-c1)//(lc//c1)+1)])) for c1 in c]
            # average number of distinct nodes in slightly overlapping windows of size w 
        w=np.array([])
    else:
        w=np.array([]); c=np.array([]); n=np.array([])
    wcn=np.asanyarray([w,c,n], dtype=object)
    return wcn

def NewNodesBiggerThanPercent(names,percent):
    steps = []
    for nickname in names:
        tf = LoadTraj(nickname+'-tf') # load trajectory data
        dte = np.concatenate([b[:-2,0] for b in tf.no]) # test states
        _,newnodes = NewNodes(dte) # calculate new nodes
        steps += [np.where(np.array(newnodes) >= percent * NUM_OF_NODES)[0][0]]
    return steps
