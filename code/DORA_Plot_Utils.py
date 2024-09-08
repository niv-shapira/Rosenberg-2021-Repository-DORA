# Imports
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import make_interp_spline

from DORA_Utils_Utils import *

#####################
####### PLOTS #######
#####################

def PlotMult(Names,fstr,parameters,frame):
    '''
    Plot multiple cross-entropy curves and new nodes for multiple mice
    Parameters:
    Names (list): list of mouse nicknames
    fstr (str): file string format for loading
    parameters (list): list of parameter ranges
    frame (int): frame size

    Returns:
    tuple: (fig, axs) matplotlib figure and axes objects
    '''
    fig, axs = plt.subplots(len(Names),1,sharex='col')
    if len(Names) <= 1:
        axs = [axs]
    axs2 = []

    color = cm.rainbow(np.linspace(0, 1, GetColorSize(parameters))) # colors for parameter combinations
    color_names = cm.rainbow(np.linspace(0, 1, len(Names))) # colors for mouse names

    # PLOT CROSS-ENTROPY FOR MODE 1
    ce_rand = -np.log2(1/3) # random policy cross-entropy
    for i,nickname in enumerate(Names):
        ces = np.load(EvalStr(fstr,nickname), allow_pickle=True) # load cross-entropy data
        fces = np.ravel(ces) # flatten ces array
        tf = LoadTraj(nickname+'-tf') # load trajectory data
        dte = np.concatenate([b[:-2,0] for b in tf.no]) # test states
        wcn = NewNodes4(dte) # calculate new nodes
        x=wcn[1,:] # steps
        y=wcn[2,:] # new nodes count

        axs2.append(axs[i].twinx())  # instantiate a second Axes that shares the same x-axis
        axs2[i].set_ylabel('New Nodes')  # we already handled the x-label with ax1
        axs2[i].plot(x,y,color='black',linewidth=2) # plot new nodes

        axs[i].hlines(y=ce_rand, xmin=0, xmax=10000, linestyle=':', color='k') # plot random policy line
        axs[i].xaxis.set_tick_params(labelbottom=True)
        axs[i].set_xscale('log')
        axs[i].set_xlabel('Steps')
        axs[i].set_ylabel('Cross Entropy')
        axs[i].set_title(nickname)

        for j,ce in enumerate(fces):
            x=np.array(range(len(ce)))*frame # calculate x-axis values
            axs[i].plot(x,ce,color=color[j],alpha=0.1,linewidth=1) # plot cross-entropy curves

    axs[0].set_xscale('log')
    axs[0].set_xlim([10,10000])
    fig.set_figheight(50/9 * len(Names))
    fig.set_figwidth(15)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return fig,axs

def PlotMinAtPercent(names,params,param_names,src_ce,fstr_sim,eff,percent=1):
    fig, axes = plt.subplots(2,2,figsize=(8, 8), gridspec_kw={'width_ratios': [0.6, 0.4], 'height_ratios': [0.5, 0.5]})
    
    ces = np.load(src_ce, allow_pickle=True)
        
    text = f"({','.join(param_names)})" +'\n' # text for parameter values 
    colors = cm.rainbow(np.linspace(0, 1, len(ces))) # colors for mouse names
    ce_rand = -np.log2(1/3) # random policy cross-entropy
    maxsteps = NewNodesBiggerThanPercent(names,percent)
            
    for ce,color,nickname,thisparams,maxstep in zip(ces,colors,names,params,maxsteps):
        # 0: NEW NODES
        tf = LoadTraj(nickname+'-tf') # load trajectory data
        dte = np.concatenate([b[:-2,0] for b in tf.no]) # test states
        x,y = NewNodes(dte) # calculate new nodes
        
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(np.array(x).min(), np.array(x).max(), 1000)
        Y_ = X_Y_Spline(X_)
        
        axes[0,0].plot(X_,Y_,color=color,linewidth=1,label=nickname) # plot new nodes
        axes[0,0].axvline(x=maxstep, linestyle=':', color=color)
        
        # 1: CEs
        x = range(len(ce)) # limit x-axis values
        y = ce
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(np.array(x).min(), np.array(x).max(), 1000)
        Y_ = X_Y_Spline(X_)
        
        axes[1,0].plot(X_,Y_,color=color,label=nickname,linewidth=1)
        axes[1,0].axvline(x=maxstep, linestyle=':', color=color)
        
        # 2: SIM
        ## mouse
        with open('outdata/'+nickname+'-Modes1', 'rb') as f: # for mouse load exploration curve from file
            _,_,_,wcn=pickle.load(f)
        x = wcn[1,:]
        y = wcn[2,:]
        axes[0,1].plot(x, y, color=color, linewidth=1, linestyle=':')
        
        ## sim
        da = np.load(EvalStr(fstr_sim,nickname), allow_pickle=True)
        wcn=NewEndNodes(da)
        x = wcn[1]
        y = wcn[2]
        axes[0,1].plot(x, y, color=color, linewidth=1, label=nickname)

        
        # 3.2: TEXT
        text = text + str(nickname) + ': '
        text = text + str(thisparams[:len(param_names)]) + '\n' # add parameter values to text
            
        
    # 0: NEW NODES   
    axes[0,0].axhline(y=0.8*126, linestyle=':', color='k', label='80%')
    #axes[0,0].set_title('New nodes as a funcion of steps')
    axes[0,0].set_ylabel('New nodes')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].set_xscale('log')
    axes[0,0].set_xlim([10,4000])
    axes[0,0].legend(loc = 'lower right')
    
    # 1: CEs
    axes[1,0].axhline(y=ce_rand, linestyle=':', color='k', label='Random') # plot random policy line
    #axes[1,0].set_title('Cross-entropy loss using parameters for minimum at ' + str(int(percent*100)) +'%')
    axes[1,0].set_ylabel('Cross-entropy loss')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].set_xscale('log')
    axes[1,0].set_xlim([10,4000])
    axes[1,0].legend(loc = 'lower right')
    
    # 2: SIM
    ## optimal
    x = [1,2,4,6,8,10,12,16,20,24,28,32,40,48,56,64]
    y = [1,2,4,6,8,10,12,16,20,24,28,32,40,48,56,64] 
    axes[0,1].plot(x, y, '-b', label='Optimal')
    ## random
    tf = LoadTraj('rw01-tf')
    da = np.concatenate([b[:-2,0] for b in tf.no])  # test states
    wcn=NewEndNodes(da)
    x = wcn[1]
    y = wcn[2]
    axes[0,1].plot(x, y, '--k', linewidth=1.5, label='Random')
    
    #axes[0,1].set_title('Efficiency using parameters for minimum at ' + str(int(percent*100)) +'%')
    axes[0,1].axhline(y=32, linestyle=':', color='k', label='32') # plot random policy line
    axes[0,1].set_ylabel('New end nodes found')
    axes[0,1].set_xlabel('End nodes visited')
    axes[0,1].legend(loc = 'lower right')
    axes[0,1].set_xlim([1,2500])
    axes[0,1].set_xscale('log')
    
    # 3: EFFICIENCY
    with open('outdata/ExplPars3', 'rb') as f:
        _,pars,_=pickle.load(f)
    k=len(RewNames) # number of rewarded animals
    U=pars[k:]
    Ua=32/U[:,0]
    maxlim = max(round(max(Ua) / 0.05) * 0.05 + 0.05, round(max(eff) / 0.05) * 0.05 + 0.05)
    minlim = min(round(min(Ua) / 0.05) * 0.05 - 0.05, round(min(eff) / 0.05) * 0.05 - 0.05)
    common_lim = [minlim,maxlim]
    ticks = np.arange(minlim, maxlim, 0.05)
    
    axes[1,1].set_xticks(ticks)
    axes[1,1].set_yticks(ticks)
    axes[1,1].set_xlim(common_lim)
    axes[1,1].set_ylim(common_lim)
    axes[1,1].set_box_aspect(1)
    axes[1,1].scatter(Ua,eff,color='green')
    axes[1,1].set_xlabel('Efficiency of mouse')
    axes[1,1].set_ylabel('Efficiency of model')
    axes[1,1].plot([minlim,maxlim],[minlim,maxlim],'--k',linewidth=1)
    
    letters = ['A', 'B', 'C', 'D']
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # ADD TITLES
    for letter, pos in zip(letters, positions):
        axes[pos].set_title(letter, loc='left', fontsize=20, pad=10)  # loc='left' places the letter to the left, pad moves it further outside the plot

    # ADD PARAMETERS
    fig.text(0.5,-0.12,text,horizontalalignment='center') # add parameter values text
    fig.set_figheight(10)
    fig.set_figwidth(20)

    return fig,axes

def ExtractValues(l):
    return l[0]

def PlotTopology4(names, fstr, params, fixed_params):
    betas,gammas,etas,deltas = params
    all_ces = np.vectorize(ExtractValues)(np.load(EvalStr(fstr,names[0]), allow_pickle=True))
    for nickname in names[1:]:
        all_ces += np.vectorize(ExtractValues)(np.load(EvalStr(fstr,nickname), allow_pickle=True))
    
    all_ces = np.divide(all_ces, len(names))

    
    fixed_indices = {'beta': fixed_params[0], 'gamma': fixed_params[1], 'eta': fixed_params[2], 'delta': fixed_params[3]}  # Adjust as needed
    
    fig,axes = plt.subplots(2,3,figsize=(12, 8))
    
    X, Y = np.meshgrid(gammas, betas[:13])
    Z = all_ces[:13, :, fixed_indices['eta'], fixed_indices['delta']]
    cntr1 = axes[0,0].contourf(X, Y, Z)
    axes[0,0].set_xlabel('gamma')
    axes[0,0].set_ylabel('beta')

    X, Y = np.meshgrid(etas, betas[:13])
    Z = all_ces[:13, fixed_indices['gamma'], :, fixed_indices['delta']]
    axes[0,1].contourf(X, Y, Z)
    axes[0,1].set_xlabel('eta')
    axes[0,1].set_ylabel('beta')
    
    X, Y = np.meshgrid(etas, gammas)
    Z = all_ces[fixed_indices['beta'], :, :, fixed_indices['delta']]
    axes[0,2].contourf(X, Y, Z)
    axes[0,2].set_xlabel('eta')
    axes[0,2].set_ylabel('gamma')
    
    X, Y = np.meshgrid(deltas, betas[:13])
    Z = all_ces[:13, fixed_indices['gamma'], fixed_indices['eta'], :]
    axes[1,0].contourf(X, Y, Z)
    axes[1,0].set_xlabel('delta')
    axes[1,0].set_ylabel('beta')

    X, Y = np.meshgrid(deltas, gammas)
    Z = all_ces[fixed_indices['beta'], :, fixed_indices['eta'], :]
    axes[1,1].contourf(X, Y, Z)
    axes[1,1].set_xlabel('delta')
    axes[1,1].set_ylabel('gamma')
    
    X, Y = np.meshgrid(deltas, etas)
    Z = all_ces[fixed_indices['beta'], fixed_indices['gamma'], :, :]
    axes[1,2].contourf(X, Y, Z)
    axes[1,2].set_xlabel('delta')
    axes[1,2].set_ylabel('eta')
    
    fig.colorbar(cntr1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    
    text = '                            (beta,gamma,eta,delta) \n Fixed parameters: ' + str([round(params[0][fixed_params[0]],1), round(params[1][fixed_params[1]],1), round(params[2][fixed_params[2]],1), round(params[3][fixed_params[3]],1)])
    fig.text(0.5,0,text,horizontalalignment='center') # add parameter values text


    return fig,axes

def PlotTopology1(names, fstr, params, fixed_params, gamma_name):
    betas,gammas,etas,deltas = params
    all_ces = np.vectorize(ExtractValues)(np.load(EvalStr(fstr,names[0]), allow_pickle=True))
    for nickname in names[1:]:
        all_ces += np.vectorize(ExtractValues)(np.load(EvalStr(fstr,nickname), allow_pickle=True))
    
    all_ces = np.divide(all_ces, len(names))

    
    fixed_indices = {'beta': fixed_params[0], 'gamma': fixed_params[1], 'eta': fixed_params[2]}  # Adjust as needed
    
    fig,axes = plt.subplots(1,3,figsize=(12, 4))
    
    X, Y = np.meshgrid(gammas, betas[:13])
    Z = all_ces[:13, :, fixed_indices['eta'],0]
    cntr1 = axes[0].contourf(X, Y, Z)
    axes[0].set_xlabel(gamma_name)
    axes[0].set_ylabel('beta')

    X, Y = np.meshgrid(etas, betas[:13])
    Z = all_ces[:13, fixed_indices['gamma'], :,0]
    axes[1].contourf(X, Y, Z)
    axes[1].set_xlabel('eta')
    axes[1].set_ylabel('beta')
    
    X, Y = np.meshgrid(etas, gammas)
    Z = all_ces[fixed_indices['beta'], :, :,0]
    axes[2].contourf(X, Y, Z)
    axes[2].set_xlabel('eta')
    axes[2].set_ylabel(gamma_name)
    
    fig.colorbar(cntr1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.2)
    
    text = '                            (beta,'+gamma_name+',eta) \n Fixed parameters: ' + str([round(params[0][fixed_params[0]],1), round(params[1][fixed_params[1]],1), round(params[2][fixed_params[2]],1)])
    fig.text(0.5,-0.05,text,horizontalalignment='center') # add parameter values text

    return fig,axes
    