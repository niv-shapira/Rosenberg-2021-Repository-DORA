# DORA Code
from DORA_Utils_Utils import *
from DORA_Mode3_Utils import *

#####################
### CALC FUNCTIONS ##
#####################

def InitSim(ma):
    '''
    Initialize simulation, return E and policy matrix, and sta = matrix of nodes and their children.
    
    Parameters:
    ma: Maze object
    
    Returns:
    tuple: (E, sta)
        E: 2D array of values
        sta: Array of nodes connected to each node
    '''
    sta = NewTransMatrix(ma)  # array of nodes connected to each node as such: index = parent node, [level, left child, right child]
    E = np.ones((len(ma.ru)+1,3))  # 2D array of values
    return E,sta

def GetPolicyState(E_state, state, prevstate, sta, beta, delta=1):
    '''
    Calculate the policy for a given state
    
    Parameters:
    E_state: Array of state values
    state: Current state
    beta: Temperature parameter for softmax
    delta: Discount factor for parent action (default=1)
    
    Returns:
    np.array: Policy probabilities for the state
    '''
    if state <= 62:
        es = deepcopy(E_state)
        i=list(sta[state]).index(prevstate)
        if beta <= 700:
            exp_beta_es = np.exp(es * beta)
            exp_beta_es[i] = delta * exp_beta_es[i]
            p_state = np.divide(exp_beta_es, exp_beta_es.sum(keepdims=True), out=np.full_like(exp_beta_es,AZERO), where=exp_beta_es.sum(keepdims=True)!=0)
        else:  # beta = infinity
            m = max(es)
            mi = np.where(es == m)[0]
            p_state = np.zeros(es.shape)
            p_state[mi] = 1/len(mi)
    elif state == 127:
        return HOME_MAT
    else:
        return TERMINAL_MAT

    return p_state

def UpdateValue(value0, value1, gamma, eta):
    '''
    Update the value of a state-action pair
    
    Parameters:
    value0: Current value
    value1: Next state value
    gamma: Discount factor
    eta: Learning rate
    
    Returns:
    float: Updated value
    '''
    return value0 + eta * (-value0 + gamma * value1)

def UpdateEState(E_state0, state0, action0, value1, gamma, eta):
    '''
    Updates the E and policy matrix. If beta > 700, assume beta=infinity
    
    Parameters:
    E_state0: Array of state values
    state0: Current state
    action0: Action taken (0 - parent, 1 - left, 2 - right)
    value1: E(s',a')
    gamma: Discount factor
    eta: Learning rate
    
    Returns:
    float: Updated value for the state-action pair
    '''
    value0 = GetValue(E_state0, action0)  # get E(s,a)
    E_state0[action0] = UpdateValue(value0, value1, gamma, eta)  # update E(s,a)
    return E_state0[action0]

def UpdateEState2(E, state0, action0, state1, action1, gamma, eta):
    '''
    Updates the E and policy matrix. If beta > 700, assume beta=infinity
    
    Parameters:
    E: Array of state-action values
    state0: Current state
    action0: Action taken (0 - parent, 1 - left, 2 - right)
    state1: Next state
    action1: Next action (-2 - simulation mode, find action, 0,1,2 - as in action0)
    gamma: Discount factor
    eta: Learning rate
    
    Returns:
    tuple: (E, action1)
        E: Updated array of state-action values
        action1: Chosen next action
    '''
    value0 = GetValue(E[state0], action0)  # get E(s,a)
    if action0 > 0:
        value_back1 = GetValue(E[state1], 0)  # get E(s',0)
        E[state1][0] = UpdateValue(value_back1, value0, gamma, eta)  # update E(s',0)

    # choose a' if simulation mode is on
    if action1 == -2:
        action1 = ChooseAction(GetPolicyState(E[state1], state1, beta))

    value1 = GetValue(E[state1], action1)  # get E(s',a')
    E[state0][action0] = UpdateValue(value0, value1, gamma, eta)  # update E(s,a)

    return E, action1

def ChooseAction(prob):
    '''
    Choose next action by policy probabilities. returns **translated action**
    
    Parameters:
    prob: Array of action probabilities
    
    Returns:
    int: Chosen action
    '''
    return int(np.random.choice(3,1,p=prob)[0])

def Simulate(nbouts, E, sta, params, mode):
    '''
    Simulate trajectories and return final E and policy matrices, and history
    
    Parameters:
    nbouts: Number of bouts (exploration until back to cage)
    E: E matrix
    sta: Parents and children matrix
    beta: Temperature parameter for softmax
    gamma: Discount factor
    eta: Learning rate
    mode: Simulation mode
    delta: Discount factor for parent action (default=1)
    
    Returns:
    tuple: (E, hist)
        E: Updated E matrix
        hist: History of states visited
    '''
    beta,gamma,eta,delta = params
    hist = []
    for i in range(nbouts):
        home = False  # has the bout ended
        j = 0
        state0 = 0
        hist += [[state0]]
        if mode == 3:
            action0 = ChooseAction(GetStateProbs(E, state0, beta, sta))
        else:
            action0 = ChooseAction(GetPolicyState(E[state0], state0, 127, sta, beta, delta))

        while (not home) and (j<1000):
            if mode == 1 or mode == 4:
                state1 = sta[state0][action0]  # state1 = next state node
                action1 = ChooseAction(GetPolicyState(E[state1], state1, state0, sta, beta, delta))
                value1 = GetValue(E[state1], action1)  # get E(s',a')
                E[state0][action0] = UpdateEState(E[state0], state0, action0, value1, gamma, eta)
                state0 = state1
                action0 = action1
            elif mode == 2:
                state1 = sta[state0][action0]  # state1 = next state node
                E, action1 = UpdateEState2(E, state0, action0, state1, -2, gamma, eta)
                state0 = state1
                action0 = action1
            elif mode == 3:
                E = UpdateE(E, state0, gamma, eta)
                action0 = ChooseAction(GetStateProbs(E, state0, beta, sta))
                state0 = sta[state0][action0]

            if state0 == 127:  # back to cage
                home = True

            hist[i] += [state0]
            j += 1

    return E, hist

def RunSim(names, params, nbouts, ma, mode, fstr=None):
    '''
    Run simulation for multiple parameter sets and save results
    
    Parameters:
    Names: List of nicknames for each parameter set
    names_params: List of parameter sets
    nbouts: Number of bouts to simulate
    ma: Maze object
    mode: Simulation mode
    fstr: Format string for saving results (optional)
    
    Returns:
    list: List of flattened histories for each parameter set
    '''
    hists = []
    for nickname, thisparams in zip(names, params):
        if len(params) < 4:
            beta, gamma, eta = thisparams
            delta = 1
        else:
            beta, gamma, eta, delta = thisparams

        if mode == 3:
            E, sta = InitSim3(ma)
        else:
            E, sta = InitSim(ma)

        _, hist = Simulate(nbouts, E, sta, thisparams, mode)
        hist = np.array(hist, dtype=object)
        fhist = np.hstack(hist)
        hists += [fhist]
        if fstr:
            np.save(EvalStr(fstr, nickname), fhist)

    return hists

#####################
### CALCULATIONS ####
#####################

def GetN32(src):
    da = np.load(src,allow_pickle=True)
    endnodes = []
    endnodes_count = 0
    for s in da:
        if s in TERMINALS:
            endnodes_count += 1
            if s not in endnodes:
                endnodes += [s]
        if len(endnodes) == 32:
            return endnodes_count
        
    return endnodes_count

def ExplainedEfficiency(eff):
    with open('outdata/ExplPars3', 'rb') as f:
        _,pars,_=pickle.load(f)
    k=len(RewNames) # number of rewarded animals
    U=pars[k:]
    Ua=32/U[:,0]
    
    expeff = []
    for modeleff,mouseeff in zip(eff,Ua):
        expeff += [modeleff / mouseeff]
    
    return expeff

def GetEfficiency(names,fstr):
    eff = []
    for nickname in names:
        eff += [32/GetN32(EvalStr(fstr,nickname))]
    
    return eff
        

def CalculateCrossEntropy2P(p1, p2):
    '''
    Calculate cross-entropy between two policies
    
    Parameters:
    p1, p2: Two policies of the same shape SxA
    
    Returns:
    float: Cross-entropy value
    '''
    ce = -np.sum(p1 * np.log2(p2, out=np.zeros_like(p2), where=(p2!=0))) / np.shape(p1)[0]
    return ce


def CalcCEs(da,mk,E,params,sta,ma,mode,frame=-1,maxsteps=-1):
    '''
    Calculate cross-entropy
    Parameters:
    da (list): data (states)
    mk (list): mask applied to action, Boolean
    E (numpy.ndarray): E-values matrix
    beta (float): inverse temperature parameter
    gamma (float): discount factor
    eta (float): learning rate
    sta (numpy.ndarray): node child matrix
    ma (object): maze object
    mode (int): simulation mode
    frame (int): frame size for CE calculation
    delta (float): delta parameter, default 1

    Returns:
    list: List of cross-entropy values
    '''
    beta,gamma,eta,delta = params
    pt=[] # predicted probabilities for the observed action
    ces = [] # list to store cross-entropy values
    for i,sn in enumerate(da): # i points to the action to be predicted
        if maxsteps != -1:
            if i == maxsteps:
                break
                
        if i+2 < len(da):
            sn1 = da[i+1] # next state
            a = GetActionFromStates(sn,sn1,ma) # get action from current and next state
            a1 = GetActionFromStates(sn1,da[i+2],ma) # get action from next and next+1 state
            
            if i == 0:
                psn = 127
            else:
                psn = da[i-1]

            if mk[i]: # if mask is True for this action
                if mode == 3:
                    probs = GetStateProbs(E,sn,beta,sta) # get probabilities for mode 3
                else:
                    probs = GetPolicyState(E[sn],sn,psn,sta,beta,delta) # get probabilities for other modes

                p = max(probs[a],AZERO) # get probability of chosen action, minimum AZERO
                pt+=[p] # add probability for the observed action to the list

                if frame != -1:
                    j = len(pt)-1
                    if (j % frame) == 0: # if we've collected enough probabilities for a frame
                        if j == 0:
                            ce = -np.log2(p) # cross-entropy for first frame
                        else:
                            pt_frame = pt[j-frame+1:j+1] # get probabilities for this frame
                            ce_frame = -np.sum(np.log2(pt_frame, out=-np.full_like(pt_frame,AZERO), where=(pt_frame!=0))) # cross-entropy for this frame
                            ce = (ces[-1] * ((len(ces)-1) * frame + 1) + ce_frame) / (len(ces) * frame + 1) # update overall cross-entropy
                        ces += [ce] # add cross-entropy to list

            # Update E-values based on mode
            if mode == 1 or mode == 4:
                value1 = GetValue(E[sn1],a1)
                E[sn][a] = UpdateEState(E[sn],sn,a,value1,gamma,eta)
            elif mode == 2:
                E,_ = UpdateEState2(E,sn,a,sn1,a1,gamma,eta)
            elif mode == 3:
                E = UpdateE(E,sn,gamma,eta)

    if frame == -1:
        ce = -np.sum(np.log2(pt, out=-np.full_like(pt,AZERO), where=(pt!=0))) / len(pt) # cross-entropy for entire sequence
        ces += [ce]

    return ces

def GetCEs(nickname,params,ma,mode,frame=-1,maxsteps=-1):
    '''
    Calculate cross entropy for a specific mouse
    Parameters:
    nickname (str): nickname of the mouse
    beta (float): inverse temperature parameter
    gamma (float): discount factor
    eta (float): learning rate
    ma (object): maze object
    mode (int): simulation mode
    frame (int): frame size for CE calculation
    delta (float): delta parameter, default 1

    Returns:
    list: List of cross-entropy values
    '''
    rew = False # unused variable
    if mode == 3:
        E,sta = InitSim3(ma) # initialize simulation for mode 3
    else:
        E,sta = InitSim(ma) # initialize simulation for other modes
    
    #endnodes = -1
    #if byendnodes:
     #   with open('outdata/ExplPars3', 'rb') as f:
      #      Names,pars,_=pickle.load(f) # number of endnodes needed to find 32 new endnodes
       #     nameindex = Names.index(nickname)
        #    endnodes = pars[nameindex,0]

    tf=LoadTraj(nickname+'-tf') # load trajectory data
    dte=np.concatenate([b[:-2,0] for b in tf.no]) # test states
    mte=np.array([True]+[True,]*(len(dte)-1)) # mask for testing, all actions OK except first
    mte[np.where(dte[:-1]>62)[0]]=False # mask for testing, eliminate end nodes
    ces = CalcCEs(dte,mte,E,params,sta,ma,mode,frame,maxsteps) # calculate cross-entropy

    return ces

def OutCEs(names,params,ma,mode,frame=-1,maxpercent=-1,fstr=None):
    '''
    Calculate and save cross-entropy for multiple mice
    Parameters:
    names (list): list of mouse nicknames
    fstr (str): file string format for saving
    parameters (list): list of parameter ranges (betas, gammas, etas, deltas)
    ma (object): maze object
    frame (int): frame size for CE calculation
    mode (int): simulation mode
    '''
    ces_mult = []
    betas,gammas,etas,deltas = params
    if maxpercent != -1:
        maxsteps = NewNodesBiggerThanPercent(names,maxpercent)
    for nickname,steps in zip(names,maxsteps):
        ces = np.ndarray(shape=(len(betas),len(gammas),len(etas),len(deltas)), dtype='object') # initialize array for cross-entropy values
        
        for b,beta in enumerate(betas):
            for g,gamma in enumerate(gammas):
                for e,eta in enumerate(etas):
                    for d,delta in enumerate(deltas):
                        ce = GetCEs(nickname,[beta,gamma,eta,delta],ma,mode,frame,steps) # calculate cross-entropy for this parameter combination
                        ces[b,g,e,d] = ce # store cross-entropy
        if fstr:
            np.save(EvalStr(fstr,nickname),ces) # save cross-entropy values to file
        ces_mult += [ces]
    
    return ces_mult

def OutCEsNamesParams(Names,params,ma,mode,frame=1,url=None):
    ces = []
    for nickname,thisparams in zip(Names,params):
        ces += [GetCEs(nickname,thisparams,ma,mode,frame)]
    if url:
        np.save(url,np.asanyarray(ces, dtype=object)) # save cross-entropy values to file
    return ces

def GetMin(ces,maxsteps=-1,indices=()):
    '''
    Find minimum cross-entropy value and its index
    Parameters:
    ces (list or numpy.ndarray): nested list or array of cross-entropy values
    maxsteps (int): maximum number of steps to consider, default -1 (all steps)
    indices (tuple): current indices in nested structure, used for recursion

    Returns:
    tuple: (min_value, min_index, min_avg_value, min_avg_index)
    '''
    min_value = float('inf')
    min_avg_value = float('inf')
    min_index = None
    min_avg_index = None
    # Check if we are at the lowest level (i.e., a list of non-lists)
    if all(isinstance(item, float) for item in ces):
        if maxsteps != -1:
            ces = ces[0:maxsteps] # limit to maxsteps if specified
        current_min = min(ces)
        current_avg = np.mean(ces)
        return current_min, indices, current_avg, indices

    # Otherwise, iterate through the nested lists
    for i, sublist in enumerate(ces):
        current_indices = indices + (i,)

        sub_min_value, sub_min_index, sub_avg_value, sub_avg_index = GetMin(sublist, maxsteps, current_indices) # recursive call

        if sub_min_value < min_value:
            min_value = sub_min_value
            min_index = sub_min_index

        if sub_avg_value < min_avg_value:
            min_avg_value = sub_avg_value
            min_avg_index = sub_avg_index

    return min_value, min_index, min_avg_value, min_avg_index

def GetStepsNumWhenCERand(nickname,params,ma,mode,minsteps,frame=1):
    rew = False # unused variable
    if mode == 3:
        E,sta = InitSim3(ma) # initialize simulation for mode 3
    else:
        E,sta = InitSim(ma) # initialize simulation for other modes

    randce = -np.log2(1/3)

    tf=LoadTraj(nickname+'-tf') # load trajectory data
    da=np.concatenate([b[:-2,0] for b in tf.no]) # test states
    mk=np.array([True]+[True,]*(len(da)-1)) # mask for testing, all actions OK except first
    mk[np.where(da[:-1]>62)[0]]=False # mask for testing, eliminate end nodes
    
    beta,gamma,eta,delta = params
    
    pt=[] # predicted probabilities for the observed action
    ces = [] # list to store cross-entropy values
    nodes = []
    for i,sn in enumerate(da): # i points to the action to be predicted
        if i+2 < len(da):
            if sn not in nodes:
                nodes += [sn]
                
            sn1 = da[i+1] # next state
            a = GetActionFromStates(sn,sn1,ma) # get action from current and next state
            a1 = GetActionFromStates(sn1,da[i+2],ma) # get action from next and next+1 state
            
            if i >= minsteps:
                if i == 0:
                    psn = 127
                else:
                    psn = da[i-1]

                if mk[i]: # if mask is True for this action
                    if mode == 3:
                        probs = GetStateProbs(E,sn,beta,sta) # get probabilities for mode 3
                    else:
                        probs = GetPolicyState(E[sn],sn,psn,sta,beta,delta) # get probabilities for other modes

                    p = max(probs[a],AZERO) # get probability of chosen action, minimum AZERO
                    pt+=[p] # add probability for the observed action to the list

                    j = len(pt)-1
                    if (j % frame) == 0: # if we've collected enough probabilities for a frame
                        if j == 0:
                            ce = -np.log2(p) # cross-entropy for first frame
                        else:
                            pt_frame = pt[j-frame+1:j+1] # get probabilities for this frame
                            ce_frame = -np.sum(np.log2(pt_frame, out=-np.full_like(pt_frame,AZERO), where=(pt_frame!=0))) # cross-entropy for this frame
                            ce = (ces[-1] * ((len(ces)-1) * frame + 1) + ce_frame) / (len(ces) * frame + 1) # update overall cross-entropy

                        if ce >= randce:
                            return i, len(nodes)
                        ces += [ce] # add cross-entropy to list

            # Update E-values based on mode
            if mode == 1 or mode == 4:
                value1 = GetValue(E[sn1],a1)
                E[sn][a] = UpdateEState(E[sn],sn,a,value1,gamma,eta)
            elif mode == 2:
                E,_ = UpdateEState2(E,sn,a,sn1,a1,gamma,eta)
            elif mode == 3:
                E = UpdateE(E,sn,gamma,eta)

    return -1, len(nodes)

def GetAvgStepsForRandCE(names,params,ma,mode,minpercent=0,frame=1):
    steps = []
    newnodes = []
    minsteps = NewNodesBiggerThanPercent(names,minpercent)
    for nickname,thisparams,minstep in zip(names,params,minsteps):
        step, newnode = GetStepsNumWhenCERand(nickname,thisparams,ma,mode,minstep,frame)
        steps += [step]
        newnodes += [newnode/126]
        
    return np.mean(steps),np.std(steps),np.mean(newnodes),np.std(newnodes)

def GetMaxSteps(nickname,percent,frame):
    '''
    Get maximum number of steps for a given percentage of new nodes
    Parameters:
    nickname (str): nickname of the mouse
    percent (float): percentage of new nodes to consider
    frame (int): frame size

    Returns:
    int: maximum number of steps
    '''
    tf = LoadTraj(nickname+'-tf') # load trajectory data
    dte = np.concatenate([b[:-2,0] for b in tf.no]) # test states
    wcn = NewNodes4(dte) # calculate new nodes
    y=wcn[2,:] # new nodes count
    argmaxsteps = np.argmax(y >= NUM_OF_NODES * percent) # find index where new nodes reach desired percentage
    x=wcn[1,:][argmaxsteps] # get corresponding step count
    return int(x / frame) # return step count in terms of frames

def GetMinArgList(Names,fstr_load,parameters,frame,percent=-1,avg=False):
    '''
    Get list of parameters that minimize average cross-entropy for each mouse
    Parameters:
    Names (list): list of mouse nicknames
    fstr_load (str): file string format for loading
    parameters (list): list of parameter ranges
    frame (int): frame size
    percent (float): percentage of new nodes to consider, default 1

    Returns:
    list: list of parameter combinations that minimize average cross-entropy
    '''
    arg_min_arr = []
    min_arr = []
    if avg:
        arg = 2
    else:
        arg = 0
        
    steps = -1
    for nickname in Names:
        if percent != -1:
            steps = GetMaxSteps(nickname,percent,frame)
        ces = np.load(EvalStr(fstr_load,nickname), allow_pickle=True) # load cross-entropy data
        res = GetMin(ces,steps)
        min_arr += [res[arg]]
        arg_min_arr += [ConvertIndicesToParameters(res[arg+1],parameters)] # get parameters that minimize average cross-entropy

    return np.asanyarray([min_arr, arg_min_arr], dtype=object)