# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:57:31 2019

@author: dyanni3
"""

#%%
import numpy as np
from scipy.linalg import circulant
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib import collections  as mc
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

#%%
def build_c(beta, N, mode = 'bipartite'):
    
    if mode=='bipartite':
        A = circulant([i%2 for i in range(N)])
        one = np.ones(N)
        A = A + np.eye(N)
        t1 = beta*(A / A.sum(axis=1))
        t2 = (1-beta)*np.eye(N)
        return(t1+t2)
    if mode=='neighbor':
        l = np.zeros(N)
        l[:2] = 1
        l[-1] =1
        A = circulant(l)
        one = np.ones(N)
        t1 = beta*(A*np.outer(1/np.dot(one,A),one))
        t2 = (1-beta)*np.eye(N)
        return(t1+t2)
    if mode=='full':
        A = np.ones((N,N))
        one = np.ones(N)
        t1 = beta*(A*np.outer(1/np.dot(one,A),one))
        t2 = (1-beta)*np.eye(N)
        return(t1+t2)
    if mode == 'star':
        A = np.eye(N)
        A[:,0] = np.ones(N)
        A[0,:] = np.ones(N)
        one = np.ones(N)
        t1 = beta*(A*np.outer(1/np.dot(one,A),one))
        t2 = (1-beta)*np.eye(N)
        return(t1+t2)

def spec(v):
    return(2*np.average(np.abs(v-.5)))
        
def W(v,c, alpha):
    return(np.sum(((v**alpha)@c)*((1-v)**alpha), axis=v.ndim-1))

"""
Here are a few small things:
1. Comments / edits on response to comment 2.
2. Can you transpose the heatmap for the star topology so beta is on the y-axis and alpha is on the x-axis?
3. Can you make alpha-beta heat maps for the viability limiting cases? One each for fully connected and bipartite topologies, please!
4. Could you also run this for viability limited to 0.5 (and again make heat maps)?
5. Also, can you make the configuration images with white backgrounds and thicker connecting lines (like the images you sent on April 1)?
"""


def W_limited(v, c, alpha, limit):
    limited = np.clip((v**alpha)@c, 0, limit)
    return(np.sum(limited*((1-v)**alpha), axis=v.ndim-1))
    
def grad_W(v,c, alpha):
    left1 = alpha*(v**(alpha-1))
    right1 = np.dot(c,(1-v)**alpha)
    t1 = left1*right1
    left2 = alpha*(1-v)**(alpha-1)
    right2 = np.dot(c,v**alpha)
    t2 = left2*right2
    return(t1-t2)
    
def W_and_grad(v, c=build_c(1,10),alpha=1):
    return( W(c,alpha,v), grad_W(c,alpha,v))
    
#%%
def spec_optimized(beta=1, N=10, mode='bipartite', alpha=1, max_iter = 100):
    bnds = tuple([(0,1) for i in range(N)])
    c = build_c(beta,N,mode)
    best = 999
    for i in range(max_iter):
        res = minimize(W, np.random.random(N), method='nelder-mead', jac=grad_W,
                   options={'disp': False, 'eps':1e-12, 'gtol':1e-9, 'ftol':1e-9},
                   args = (c,alpha), bounds=bnds)
        if res.fun<best:
            best = res.fun
            v = res.x
    else:
        res = minimize(W, np.random.random(N), method='L-BFGS-B', jac=grad_W,
                   options={'disp': True}, args = (c,alpha), bounds=bnds)
        pass
    spec = np.average([2*(max([vi, (1-vi)])-.5) for vi in v])
    #spec = np.average(v)
    return(spec,best,v)
    
#%% parameter sweep
def make_plot(mode='bipartite'):
    alphas = np.linspace(0.01,1.5,50)
    betas = np.linspace(0,1,4)
    specs = [[] for i in range(10)]
    for i,beta in enumerate(betas):
        print(i)
        for j,alpha in enumerate(alphas):
            print(j)
            s,res = spec_optimized(beta=beta, alpha=alpha, mode = mode ,N=10)
            specs[i].append(s)
            
    fig = plt.figure(figsize = (10,10))
    plt.xlabel("Specialization Power",fontsize = 14)
    plt.ylabel("Specialization",fontsize=14)
    plt.title("%s network"%mode,fontsize=14)
    for i in range(4):
        plt.plot(alphas,specs[i],marker='o',lw=0, label = betas[i])
    plt.legend()
    return(fig)
#%%
def gradient_descent(eps, v0, beta=1, N=10, mode='bipartite', alpha=1, max_iter = 1000):
    ws = []
    c = build_c(beta,N,mode)
    for i in range(max_iter):
        v0 = v0 + eps*grad_W(v0,c,alpha)
        v0[v0>1]=1
        v0[v0<0]=0
        ws.append(W(v0,c,alpha))
    return(v0,ws)
    
def spec(v):
    return(np.average([2*(max([vi, (1-vi)])-.5) for vi in v]))

def make_network_for_plotting(N=10, mode='bipartite'):
    theta_step = 2*np.pi/(N)
    thetas = [theta_step*i for i in range(N)]
    if mode == 'star':
        theta_step = 2*np.pi/(N-1)
        thetas = [theta_step*i for i in range(N-1)]
    pts = []
    if mode == 'star':
        pts.append((0,0))
    for theta in thetas:
        pts.append( (np.cos(theta), np.sin(theta)) )

    lines = []
    for i,pt in enumerate(pts):
        if mode == 'star':
            opt = ((0,0))
            lines.append([pt, opt])
        for j,opt in enumerate(pts):
            if mode == 'full':
                lines.append([pt,opt])
            elif mode =='bipartite':
                if ((i+j)%2==1):
                    lines.append([pt, opt])
            elif mode == 'neighbor':
                if (j==(i+1)%N):
                    lines.append([pt, opt])

    return(np.array(pts), lines)

def make_colorized(v, N=10, mode='bipartite'):
    pts, lines = make_network_for_plotting(N=N, mode=mode)
    fig, ax = plt.subplots(figsize = (10,10))
    lc = mc.LineCollection(lines, colors='k', linewidths = 3.5)
    ax.add_collection(lc)
    ax.scatter(pts[:,0],pts[:,1],
        c=[plt.cm.bwr(1-vi) for vi in v], s=1500, edgecolor='k')
    ax.autoscale()
    fig2,ax2 = plt.subplots()
    cb = ax2.imshow(np.stack([np.linspace(min(v), max(v), 50) for i in range(6)]).T,
        cmap=plt.cm.bwr)
    fig2.colorbar(cb, ax=ax2)
    return(fig,fig2)

def make_mean_field_cartoon_networks():
    home = os.getcwd()
    pts = [(i,j) for i in range(5) for j in range(5)]
    lines = []
    for i in range(5):
        for j in range(5):
            for inc in [-1, 0, 1]:
                lines.append([(i,j),(i,j + inc)]) 
                lines.append([(i,j),(i + inc ,j)])
    pts = np.array(pts)
    color_choices = ['r', 'b']
    c = [color_choices[i % 2] for i in range(25)] 
    fig, ax = plt.subplots(figsize = (10,10))
    lc = mc.LineCollection(lines, colors='k', linewidths = 2)
    ax.add_collection(lc)
    ax.scatter(pts[:,0],pts[:,1], c = c, s=1500, edgecolor='k')
    ax.autoscale()
    fig.savefig(home+ r'\cartoons\f1Xp5z4.pdf')

    color_choices = ['r', 'b' , 'r']
    c = [color_choices[i % 3] for i in range(25)] 
    lc = mc.LineCollection(lines, colors='k', linewidths = 2)
    fig, ax = plt.subplots(figsize = (10,10))
    ax.add_collection(lc)
    ax.scatter(pts[:,0],pts[:,1], c = c, s=1500, edgecolor='k')
    ax.autoscale()
    fig.savefig(home + r'\cartoons\fp5Xp66z4.pdf')

    bpts = [(i,j) for i in range(0,10,2) for j in range(-2,3)]
    rpts = [(i,0) for i in range(1, 11, 2)] 
    lines = []
    for i in range(1, 11, 2):
        for j in range(-2,3):
            lines.append([(i,0), (i-1,j)])
            if i!=9:
                lines.append([(i,0), (i+1,j)])
    bpts = np.array(bpts)
    rpts = np.array(rpts)
    fig, ax = plt.subplots(figsize = (10,10))
    lc = mc.LineCollection(lines, colors='k', linewidths = 2)
    ax.add_collection(lc)
    ax.scatter(bpts[:,0],bpts[:,1], c = 'b', s=1500, edgecolor='k')
    ax.scatter(rpts[:,0],rpts[:,1], c = 'r', s=1500, edgecolor='k')
    ax.autoscale()
    fig.savefig(home + r'\cartoons\f1Xp16z10.pdf')


def evo(pop, c, alpha, nsteps, limited=False):
    N = pop.shape[0]
    for t in range(nsteps):
        mask = np.random.choice([0,1], p=[.98, .02], size=(N, 10))
        mut = mask*(.1*(np.random.random((N,10))-.5))
        pop += mut
        pop = np.clip(pop,0,1)
        if limited==False:
            fits = W(pop, c, alpha)
        else:
            fits = W_limited(pop, c, alpha, limited)
        inds = np.random.choice(np.arange(N), p=fits/np.sum(fits),
         replace=True, size=N)
        pop = pop[inds]
        #print(f"ep: {t}, fit: {np.average(fits)}")
    return(pop)

def make_fig4_ab_data(mode, nsteps, popsize):
    alphas = np.linspace(.5,1.5,11)
    c = build_c(1, 10, mode=mode)
    spec_means = []
    spec_stds = []
    for alpha in alphas:
        print(alpha)
        pop = np.random.rand(popsize, 10)
        pop = evo(pop, c, alpha, nsteps, limited=False)
        spec_means.append(np.average([spec(v) for v in pop]))
        spec_stds.append(np.std([spec(v) for v in pop]))
    return(spec_means, spec_stds)

def make_fig4_ab_figure(title, sm, ss):
    with plt.style.context('seaborn-white'):
        alphas = np.linspace(.5, 1.5, 11)
        fig = plt.figure(figsize=(8,6))
        plt.vlines(1.0, 0, 1, colors='gold', linestyles='dashed', lw=2)
        plt.errorbar(alphas, sm, yerr=ss, ecolor='k', elinewidth=1.8, capsize=2, marker='o', markersize=10, color='b', lw = 0)
        plt.xlabel(r'$\alpha$', fontsize=18)
        plt.ylabel('Specialization', fontsize=18)
        plt.grid(lw=.2)
        plt.xticks([round(x, 2) for x in alphas], fontsize=18)
        plt.yticks(np.arange(1, 11)/10, fontsize=18)
        plt.ylim(0,1)
        plt.title(title, fontsize=18)




def make_hmap(mode, nsteps, popsize, limited):
    alphas = np.linspace(.1,1,10)
    betas = np.linspace(0,1,11)
    hm = np.zeros((10,11))
    print(f"mode: {mode} limit: {limited}")
    for i,alpha in tqdm(enumerate(alphas)):
        for j,beta in enumerate(betas):
            c = build_c(beta, 10, mode=mode)
            #print(f"mode:{mode} limit:{limited} alpha: {alpha}, beta: {beta}")
            pop = np.random.rand(popsize,10)
            pop = evo(pop, c, alpha, nsteps, limited)
            hm[i,j] = np.average([spec(v) for v in pop])
    return((mode, limited, hm))


def mh(x):
    return(make_hmap(*x))

def make_hmaps_par(modes, limits, nsteps=2000, popsizes=10000):
    situations = []
    for mode in modes:
        for lim in limits:
            situations.append((mode,nsteps, popsizes, lim))
    with mp.Pool(6) as pool:
        res = pool.starmap(make_hmap, situations)
        return([i for i in res])

def make_hmap_figure_from_data(fname):
    df = pd.read_csv(fname).drop('Unnamed: 0', axis=1)
    fig, ax = plt.subplots(figsize=(6,8))
    ax.imshow(np.rot90(df))
    linspace = np.linspace(1,0,11)
    rounded = [round(x,2) for x in linspace]
    ax.set_yticklabels(rounded, fontsize=14)
    ax.set_yticks(np.arange(11))
    linspace = np.linspace(.1,1,10)
    rounded = [round(x,2) for x in linspace]
    ax.set_xticklabels(rounded, fontsize=14)
    ax.set_xticks(np.arange(10))
    ax.set_xlabel(r"$\alpha$", fontsize=18)
    ax.set_ylabel(r"$\beta$", fontsize=18)
    to_save_fname = fname.split('.')
    if len(to_save_fname)>2:
        to_save_fname = to_save_fname[0] + to_save_fname[1]
    else:
        to_save_fname = to_save_fname[0]
    to_save_fname = to_save_fname + "_figure.pdf"
    fig.savefig(to_save_fname)

def star_analysis_plot():
    n = np.linspace(1,20,1000)
    num = 2*n**2
    denom = num + 2*n + 8
    res = -.5*np.log2(num/denom)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(n, res, 'b', lw=3)
    ax.grid(lw=.8, c='gray')
    ax.set_xlabel("Number of individuals", fontsize=16)
    ax.set_ylabel(r"$\alpha_{max}$", fontsize=16)
    ax.set_ylim(0,1)
    ax.set_xticklabels(ax.get_xticks(), fontsize=14)
    ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=14)
    return fig

if __name__ == '__main__':
    modes = ['bipartite', 'full', 'neighbor', 'star']
    limits = [0.2, 0.5, False]
    results = make_hmaps_par(modes, limits)
    for result in results:
        fname = f"{result[0]}_{result[1]}.csv"
        df = pd.DataFrame(result[2])
        df.to_csv(fname)
        make_hmap_figure_from_data(fname)



