import numpy as np
from MM_Maze_Utils import *
from MM_Traj_Utils import TransMatrix

#####################
###### MODE 3 #######
#####################

def NewTransMatrix(ma):
    '''
    Add node 127 to trans matrix
    '''
    tra = TransMatrix(ma)
    tra = np.append(tra,[[-1,0,0]],axis=0)
    return tra

def TranslateAction(action):
    '''
    Translate action from original action code to our action code
    '''
    if (action == -1):
        return -1
    else:
        return (action + 1) % 3

def GetActionFromStates(state0,state1,ma):
    return TranslateAction(StepType2(state0, state1, ma)) # step type from MM_Maze_Utils

def InitSim3(ma):
    '''
    initialize simulation, return E and policy matrix, and sta = matrix of nodes and their children.
    '''
    sta=NewTransMatrix(ma) # array of nodes connected to each node as such: index = parent node, [level, left child, right child]
    E = np.array([1] * len(sta), dtype='float') # 1D array of values
    return E,sta

def ExpBeta(value,beta):
    return np.exp(beta * value)

def GetStateProbs(E,state,beta,sta):
    probs = np.array([ExpBeta(E[s],beta) for s in sta[state]])
    sum_probs = np.sum(probs)
    return probs / sum_probs

def UpdateEState(value,eta):
    return value - eta * value

def AttenuateE(E,gamma):
    E = E + gamma * E # apply the attenuation
    E = np.clip(E, 0, 1) # clip the values to ensure they don't exceed 1
    return E

def UpdateE(E,state,gamma,eta):
    E[state] = UpdateEState(E[state],eta)
    E = AttenuateE(E,gamma)
    return E