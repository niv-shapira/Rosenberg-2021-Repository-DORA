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
    # Create transition matrix showing connections between nodes
    # For each node, it stores [parent, left child, right child]
    sta = NewTransMatrix(ma)
    
    # Initialize E-values matrix with ones
    # Shape is (number of nodes + 1, 3) where 3 corresponds to the three possible actions
    E = np.ones((len(ma.ru)+1,3))
    
    return E,sta

def GetPolicyState(E_state, state, prevstate, sta, beta, delta=1):
    '''
    Calculate the policy for a given state
    
    Parameters:
    E_state: Array of state values
    state: Current state
prevstate: Previous state visited
    sta: Array of nodes connected to each node
    beta: Temperature parameter for softmax
    delta: Discount factor for parent action (default=1)
    
    Returns:
    np.array: Policy probabilities for the state
    '''
    if state <= 62:
        # Make a deep copy of the state values to avoid modifying the original
        es = deepcopy(E_state)
        
        # Find index of prevstate in the state's connections
        i = list(sta[state]).index(prevstate)
        
        if beta <= 700:
            # Calculate softmax probabilities with finite temperature
            # Multiply each value by beta (inverse temperature)
            exp_beta_es = np.exp(es * beta)
            
            # Apply delta to the parent action (makes returning less/more likely)
            exp_beta_es[i] = delta * exp_beta_es[i]
            
            # Normalize to get probabilities, handling potential division by zero
            p_state = np.divide(exp_beta_es, exp_beta_es.sum(keepdims=True), 
                               out=np.full_like(exp_beta_es,AZERO), 
                               where=exp_beta_es.sum(keepdims=True)!=0)
        else:
            # For very high beta (approaching infinity), just pick the max value(s)
            m = max(es)
            mi = np.where(es == m)[0]
            
            # Initialize with zeros and distribute probability equally among max values
            p_state = np.zeros(es.shape)
            p_state[mi] = 1/len(mi)
    elif state == 127:
        # Special case: at home state
        return HOME_MAT
    else:
        # Special case: at terminal state
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
    # Temporal difference learning update
    # New value = old value + learning rate * (discounted next value - old value)
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
    # Get current value for the state-action pair
    value0 = GetValue(E_state0, action0)
    
    # Update the value using temporal difference learning
    E_state0[action0] = UpdateValue(value0, value1, gamma, eta)
    
    # Return the updated value
    return E_state0[action0]

def ChooseAction(prob):
    '''
    Choose next action by policy probabilities. returns **translated action**
    
    Parameters:
    prob: Array of action probabilities
    
    Returns:
    int: Chosen action
    '''
    # Randomly choose an action based on the probability distribution
    return int(np.random.choice(3,1,p=prob)[0])

def Simulate(nbouts, E, sta, params):
    '''
    Simulate trajectories and return final E and policy matrices, and history
    
    Parameters:
    nbouts: Number of bouts (exploration until back to cage)
    E: E matrix
    sta: Parents and children matrix
params: List containing [beta, gamma, eta, delta]
    beta: Temperature parameter for softmax
    gamma: Discount factor
    eta: Learning rate
        delta: Discount factor for parent action
    
    Returns:
    tuple: (E, hist)
        E: Updated E matrix
        hist: History of states visited
    '''
    # Unpack parameters
    beta,gamma,eta,delta = params
    
    # Initialize history list to store trajectories
    hist = []
    
    # Simulate nbouts exploration bouts
    for i in range(nbouts):
        home = False  # Track if the animal has returned home
        j = 0  # Step counter to prevent infinite loops
        
        # Start at the root node (state 0)
        state0 = 0
        hist += [[state0]]  # Add initial state to history
        
        # Choose first action from root node
        # Previous state is set to 127 (home) for the root node
        action0 = ChooseAction(GetPolicyState(E[state0], state0, 127, sta, beta, delta))

        # Continue exploration until returning home or reaching step limit
        while (not home) and (j<1000):
            # Determine next state based on current state and action
            state1 = sta[state0][action0]
            
            # Choose next action based on new state
            action1 = ChooseAction(GetPolicyState(E[state1], state1, state0, sta, beta, delta))
            
            # Get value of next state-action pair
            value1 = GetValue(E[state1], action1)
            
            # Update E-value for current state-action pair
            E[state0][action0] = UpdateEState(E[state0], state0, action0, value1, gamma, eta)
            
            # Move to next state and action
            state0 = state1
            action0 = action1

            # Check if we've returned home
            if state0 == 127:
                home = True

            # Record state in history
            hist[i] += [state0]
            
            # Increment step counter
            j += 1

    return E, hist

def RunSim(names, params, nbouts, ma, fstr=None):
    '''
    Run simulation for multiple parameter sets and save results
    
    Parameters:
    names: List of nicknames for each parameter set
    params: List of parameter sets [beta, gamma, eta, delta]
    nbouts: Number of bouts to simulate
    ma: Maze object
        fstr: Format string for saving results (optional)
    
    Returns:
    list: List of flattened histories for each parameter set
    '''
    # Initialize list to store histories
    hists = []
    
    # Loop through each parameter set and its nickname
    for nickname, thisparams in zip(names, params):
        # Unpack parameters
        beta, gamma, eta, delta = thisparams

        # Initialize simulation
        E, sta = InitSim(ma)

        # Run simulation and get resulting history
        _, hist = Simulate(nbouts, E, sta, thisparams)
        
        # Convert history to numpy array
        hist = np.array(hist, dtype=object)
        
        # Flatten history for analysis
        fhist = np.hstack(hist)
        
        # Add to list of histories
        hists += [fhist]
        
        # Save history if requested
        if fstr:
            np.save(EvalStr(fstr, nickname), fhist)

    return hists

#####################
### CALCULATIONS ####
#####################

def GetN32(src):
    '''
    Count visits to terminal nodes until all 32 unique terminal nodes are found
    
    Parameters:
    src (str): Source file path for trajectory data
    
    Returns:
    int: Count of terminal node visits required to find all 32 unique terminal nodes
    '''
    # Load trajectory data
    da = np.load(src, allow_pickle=True)
    
    # Initialize list to track unique terminal nodes found
    endnodes = []
    # Counter for terminal node visits
    endnodes_count = 0
    
    # Iterate through states in trajectory
    for s in da:
        # Check if current state is a terminal node
        if s in TERMINALS:
            endnodes_count += 1
            # Add to unique terminal nodes if not already found
            if s not in endnodes:
                endnodes += [s]
            # If all 32 terminal nodes found, return count
            if len(endnodes) == 32:
                return endnodes_count
    
    # Return total count if not all terminal nodes were found
    return endnodes_count

def ExplainedEfficiency(eff):
    '''
    Calculate explained efficiency relative to mouse efficiency
    
    Parameters:
    eff (list): List of model efficiencies
    
    Returns:
    list: Explained efficiencies (model efficiency / mouse efficiency)
    '''
    # Load parameters from file
    with open('outdata/ExplPars3', 'rb') as f:
        _,pars,_=pickle.load(f)
    
    # Number of rewarded animals
    k=len(RewNames)
    
    # Get parameters for unrewarded animals
    U=pars[k:]
    
    # Calculate efficiency of real mice (32/number of terminal visits needed)
    Ua=32/U[:,0]
    
    # Calculate explained efficiency (model efficiency / mouse efficiency)
    expeff = []
    for modeleff,mouseeff in zip(eff,Ua):
        expeff += [modeleff / mouseeff]
    
    return expeff

def GetEfficiency(names,fstr):
    '''
    Calculate efficiency for multiple models
    
    Parameters:
    names (list): List of model nicknames
    fstr (str): Format string for loading trajectory data
    
    Returns:
    list: Efficiencies for each model
    '''
    # Initialize list to store efficiencies
    eff = []
    
    # Calculate efficiency for each model
    for nickname in names:
        # Efficiency = 32 / number of terminal visits needed to find all 32 nodes
        eff += [32/GetN32(EvalStr(fstr,nickname))]
    
    return eff


def CalcCEsStep(da,mk,E,params,sta,ma):
    '''
    Calculate cross-entropy step by step
    
    Parameters:
    da (list): List of states
    mk (list): Mask applied to action, Boolean
    E (numpy.ndarray): E-values matrix
    params (list): List containing [beta, gamma, eta, delta]
    sta (numpy.ndarray): Node child matrix
    ma (object): Maze object
    
    Returns:
    tuple: (ces, ces_bouts)
        ces: List of cross-entropy values
        ces_bouts: List of cross-entropy values per bout
    '''
    # Unpack parameters
    beta,gamma,eta,delta = params
    
    # Initialize lists to track probabilities and cross-entropy
    pt=[]  # Predicted probabilities for all observed actions
    pt_bout = []  # Predicted probabilities for current bout
    ces = []  # Overall cross-entropy
    boutnum = 0  # Current bout number
    ces_bouts = []  # Cross-entropy per bout
    
    # Iterate through states
    for i,sn in enumerate(da):
        # Check if this is the home state (end of bout)
        if sn == 127:
            if pt_bout:
                # Calculate cross-entropy for the completed bout
                # Higher values mean worse prediction
                ces_bouts += [-np.sum(np.log2(pt_bout, 
                                            out=-np.full_like(pt_bout,AZERO), 
                                            where=(pt_bout!=0))) / len(pt_bout)]
                pt_bout = []  # Reset for next bout
                boutnum += 1  # Increment bout counter
            continue
        
        # Only process if we have enough states ahead to look at next actions
        if i+2 < len(da):
            sn1 = da[i+1]  # Next state
            # Determine action taken to go from current to next state
            a = GetActionFromStates(sn,sn1,ma)
            # Determine action taken to go from next to next+1 state
            a1 = GetActionFromStates(sn1,da[i+2],ma)
            
            # Get previous state
            psn = da[i-1]

            # If this action should be included in analysis (based on mask)
            if mk[i]:
                # Get action probabilities
                probs = GetPolicyState(E[sn],sn,psn,sta,beta,delta)

                # Get probability of the action that was actually taken
                p = max(probs[a],AZERO)  # Use minimum value AZERO to avoid log(0)
                pt+=[p]  # Add to overall probabilities
                pt_bout += [p]  # Add to current bout probabilities

            # Update E-values based on observed transitions
            value1 = GetValue(E[sn1],a1)
            E[sn][a] = UpdateEState(E[sn],sn,a,value1,gamma,eta)

        # Calculate running cross-entropy
        ces += [-np.sum(np.log2(pt, out=-np.full_like(pt,AZERO), where=(pt!=0))) / len(pt)]
    
    return ces, ces_bouts

def CalcCEs(da,mk,E,params,sta,ma):
    '''
    Calculate cross-entropy
    
    Parameters:
    da (list): List of states
    mk (list): Mask applied to action, Boolean
    E (numpy.ndarray): E-values matrix
    params (list): List containing [beta, gamma, eta, delta]
    sta (numpy.ndarray): Node child matrix
    ma (object): Maze object
    
    Returns:
    list: List of cross-entropy values
    '''
    # Unpack parameters
    beta,gamma,eta,delta = params
    
    # Initialize variables
    pt=[]  # Predicted probabilities for observed actions
    seg=5  # Number of segments to divide data into
    ces=np.zeros(seg)  # Cross-entropy for each segment
    ces_count = np.zeros(seg)  # Count of bouts in each segment
    boutnum=-1  # Current bout number (start at -1, incremented at first home state)
    
    # Iterate through states
    for i,sn in enumerate(da):
        # Check if this is the home state (end of bout)
        if sn == 127:
            if boutnum > -1 and pt:
                # Calculate cross-entropy for the completed bout
                # Add to appropriate segment based on bout number
                ces[boutnum % seg] -= np.sum(np.log2(pt, 
                                                    out=-np.full_like(pt,AZERO), 
                                                    where=(pt!=0))) / len(pt)
                ces_count[boutnum % seg] += 1  # Increment count for this segment
                pt = []  # Reset for next bout
            
            boutnum += 1  # Increment bout counter
            continue
        
        # Only process if we have enough states ahead to look at next actions
        if i+2 < len(da):
            sn1 = da[i+1]  # Next state
            # Determine action taken to go from current to next state
            a = GetActionFromStates(sn,sn1,ma)
            # Determine action taken to go from next to next+1 state
            a1 = GetActionFromStates(sn1,da[i+2],ma)
            
            # Get previous state
            psn = da[i-1]

            # If this action should be included in analysis (based on mask)
            if mk[i]:
                # Get action probabilities
                probs = GetPolicyState(E[sn],sn,psn,sta,beta,delta)

                # Get probability of the action that was actually taken
                p = max(probs[a],AZERO)  # Use minimum value AZERO to avoid log(0)
                pt+=[p]  # Add to probabilities list

            # Update E-values based on observed transitions
            value1 = GetValue(E[sn1],a1)
            E[sn][a] = UpdateEState(E[sn],sn,a,value1,gamma,eta)

    # Return average cross-entropy across segments
    return [np.mean(ces / ces_count)]

def GetCEs(nickname,params,ma,test=True):
    '''
    Calculate cross entropy for a specific mouse
    
    Parameters:
    nickname (str): Nickname of the mouse
    params (list): List containing [beta, gamma, eta, delta]
    ma (object): Maze object
    test (bool): Whether to run in test mode (default=True)
    
    Returns:
    list or tuple: List of cross-entropy values or tuple (ces, ces_bouts) if test=False
    '''
    # Initialize simulation
    E,sta = InitSim(ma)

    # Load trajectory data for this mouse
    tf=LoadTraj(nickname+'-tf')
    
    # Extract state sequence from trajectory
    # Each element of tf.no is a bout, and each bout is an array of [state, X]
    # We only need the state (first column)
    dte=np.concatenate([b[:,0] for b in tf.no])
    
    # Create mask for actions to include in analysis
    # All actions are included except the very first one
    mte=np.array([True]+[True,]*(len(dte)-1))
    
    # Exclude actions from terminal nodes (states > 62)
    mte[np.where(dte[:-1]>62)[0]]=False
    
    # Calculate cross-entropy based on test flag
    if test:
        ces = CalcCEs(dte,mte,E,params,sta,ma)
    else:
        ces, ces_bouts = CalcCEsStep(dte,mte,E,params,sta,ma)
        return ces, ces_bouts
    
    return ces

def OutCEs(names,params,ma,fstr=None):
    '''
    Calculate and save cross-entropy for multiple mice
    
    Parameters:
    names (list): List of mouse nicknames
    params (list): List containing [betas, gammas, etas, deltas] of parameter ranges
    ma (object): Maze object
    fstr (str): File string format for saving (optional)
    
    Returns:
    list: List of cross-entropy values for all mice
    '''
    # Initialize list to store results
    ces_mult = []
    
    # Unpack parameters into separate arrays for grid search
    betas,gammas,etas,deltas = params
    
    # Process each mouse
    for nickname in names:
        # Create multidimensional array to store cross-entropy for each parameter combination
        ces = np.ndarray(shape=(len(betas),len(gammas),len(etas),len(deltas)), dtype='object')
        
        # Grid search over all parameter combinations
        for b,beta in enumerate(betas):
            for g,gamma in enumerate(gammas):
                for e,eta in enumerate(etas):
                    for d,delta in enumerate(deltas):
                        # Calculate cross-entropy for this parameter set
                        ce = GetCEs(nickname,[beta,gamma,eta,delta],ma)
                        # Store result in appropriate position in array
                        ces[b,g,e,d] = ce
        
        # Save results if requested
        if fstr:
            np.save(EvalStr(fstr,nickname),ces)
        
        # Add to overall results
        ces_mult += [ces]
    
    return ces_mult

def OutCEsNamesParams(Names,params,ma,url=None):
    '''
    Calculate and optionally save cross-entropy for multiple mice with specific parameters
    
    Parameters:
    Names (list): List of mouse nicknames
    params (list): List of parameter sets [beta, gamma, eta, delta]
    ma (object): Maze object
    url (str): Save location (optional)
    
    Returns:
    tuple: (ces, ces_bouts) Lists of cross-entropy values
    '''
    # Initialize lists to store results
    ces = []
    ces_bouts = []
    
    # Process each mouse with its corresponding parameter set
    for nickname,thisparams in zip(Names,params):
        # Calculate cross-entropy (test=False for detailed results)
        ce, ce_bouts = GetCEs(nickname,thisparams,ma,test=False)
        # Store results
        ces += [ce]
        ces_bouts += [ce_bouts]
    
    # Save results if requested
    if url:
        np.save(url,np.asanyarray([ces,ces_bouts], dtype=object))
    
    return ces, ces_bouts

def GetMin(ces, indices=()):
    '''
    Find minimum cross-entropy value and its index
    
    Parameters:
    ces (list or numpy.ndarray): Nested list or array of cross-entropy values
    indices (tuple): Current indices in nested structure, used for recursion (default=())
    
    Returns:
    tuple: (min_value, min_index, min_avg_value, min_avg_index)
        min_value: Minimum cross-entropy value
        min_index: Indices of minimum value
        min_avg_value: Minimum average cross-entropy value
        min_avg_index: Indices of minimum average value
    '''
    # Initialize variables to track minima
    min_value = float('inf')
    min_avg_value = float('inf')
    min_index = None
    min_avg_index = None
    
    # Base case: reached the bottom level (list of float values)
    if all(isinstance(item, float) for item in ces):
        current_min = min(ces)  # Find minimum value
        current_avg = np.mean(ces)  # Calculate average
        return current_min, indices, current_avg, indices

    # Recursive case: iterate through nested lists
    for i, sublist in enumerate(ces):
        # Add current index to the accumulated indices
        current_indices = indices + (i,)

        # Recursively find minima in sublists
        sub_min_value, sub_min_index, sub_avg_value, sub_avg_index = GetMin(sublist, current_indices)

        # Update minimum value and its index if we found a better one
        if sub_min_value < min_value:
            min_value = sub_min_value
            min_index = sub_min_index

        # Update minimum average and its index if we found a better one
        if sub_avg_value < min_avg_value:
            min_avg_value = sub_avg_value
            min_avg_index = sub_avg_index

    return min_value, min_index, min_avg_value, min_avg_index
    
def GetMinArgList(Names, fstr_load, parameters, avg=False):
    '''
    Get list of parameters that minimize average cross-entropy for each mouse
    
    Parameters:
    Names (list): List of mouse nicknames
    fstr_load (str): File string format for loading
    parameters (list): List of parameter ranges
    avg (bool): Whether to use average values (default=False)
    
    Returns:
    numpy.ndarray: Array containing [min_values, optimal_parameters]
    '''
    arg_min_arr = []  # Will store optimal parameter sets
    min_arr = []      # Will store minimum cross-entropy values
    
    # Determine whether to use minimum value or average value
    if avg:
        arg = 2  # Index for average values in GetMin results
    else:
        arg = 0  # Index for minimum values in GetMin results
        
    for nickname in Names:
        # Load cross-entropy data for this mouse
        ces = np.load(EvalStr(fstr_load,nickname), allow_pickle=True)
        
        # Find minimum cross-entropy and corresponding parameters
        res = GetMin(ces)
        
        # Store the minimum value
        min_arr += [res[arg]]
        
        # Convert parameter indices to actual parameter values and store
        arg_min_arr += [ConvertIndicesToParameters(res[arg+1],parameters)]

    # Return both arrays as a single numpy array
    return np.asanyarray([min_arr, arg_min_arr], dtype=object)

def GetLeftRightBias(nickname, ma):
    '''
    Calculate left/right bias in trajectory
    
    Parameters:
    nickname (str): Mouse nickname
    ma (object): Maze object
    
    Returns:
    float: Ratio of left choices to total left+right choices
    '''
    # Load trajectory data for this mouse
    tf=LoadTraj(nickname+'-tf')
    
    # Extract state sequences from trajectory data
    # b[:-2,0] excludes the last two entries of each bout (typically home state)
    da=np.concatenate([b[:-2,0] for b in tf.no])
    
    lrsteps = 0  # Total number of left/right choices (non-parent actions)
    lsteps = 0   # Number of left choices (action = 1)
    
    # Iterate through states
    for i,sn in enumerate(da):
        if i+1 < len(da):
            sn1 = da[i+1]  # Next state
            
            # Determine which action was taken to move from sn to sn1
            a = GetActionFromStates(sn,sn1,ma)

            # If action is not parent (0), count it
            if a != 0:
                lrsteps += 1  # Increment total left/right count
                lsteps += a % 2  # a % 2 equals 1 for left (a=1) and 0 for right (a=2)

    # Return proportion of left choices
    return lsteps / lrsteps