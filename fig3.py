import numpy as np
from matplotlib import pyplot as plt
    
#%%

def spec(v):
    s = 0
    for elem in v:
        s+=2*abs(elem-.5)
    return(s/len(v))
    
def make_q_rand(N,M):
    q = np.zeros(N**2);
    q[np.random.choice(np.arange(N**2),replace=False,size=M)]=1
    q = q.reshape((N,N))
    q = q+q.T
    q = np.clip(q+np.eye(N),0,1)
    q = q/np.sum(q,axis=0)
    return(q.T)
    
def make_q_neighbor(N):
    row = np.zeros(N)
    row[:3]=1
    A = np.stack([np.roll(row,i) for i in range(N)])
    A = np.roll(A,1,axis=0)
    q = A/np.sum(A,axis=1)
    return(q)
    
def make_q_bipartite(N):
    row = np.zeros(N)
    row[::2]=1
    A = np.stack([np.roll(row,i) for i in range(N)])
    q = A/np.sum(A,axis=1)
    return(q)
    
def W(alpha, beta, q, v):
    c = beta*q + (1-beta)*np.eye(q.shape[0])
    return((v**alpha)@(c@(1-v)**alpha))
    
    
def grad(alpha,beta,q,v):
    c = beta*q + (1-beta)*np.eye(q.shape[0])
    left = (v**(alpha-1))*(c@((1-v)**alpha))
    right = ((1-v)**(alpha-1))*(c.T@(v**alpha))
    return(left-right)

def H(alpha, beta, q, v):
    c = beta*q + (1-beta)*np.eye(q.shape[0])
    o1 = np.outer(v**(alpha-1),(1-v)**(alpha-1))
    o3 = np.outer(c@((1-v)**alpha),v**(alpha-2))
    o4 = np.outer((1-v)**(alpha-2),c.T@(v**alpha))
    return( (-alpha**2)*(c*o1 + c.T*o1.T) + (alpha*(alpha-1))*(o3+o4))
    
def Newton_Raphson(alpha,beta,q,v0=None):
    N = q.shape[0]
    one = np.ones(N)
    if v0 is None:
        v = .5*one
    else:
        v=v0
    for t in range(10):
        v = v - np.linalg.inv(H(alpha,beta,q,v))@grad(alpha,beta,q,v)
        v = np.clip(v,0,1)
    return(np.round(v,3))
    
def descent(alpha,beta,q,v=None):
    N = q.shape[0]
    one = np.ones(N)
    if v is None:
        v = .5*one
    for t in range(1000):
        v = v + .01*grad(alpha,beta,q,v)
        v = np.clip(v,0,1)
    return(v)
    
#%%
hmap_rand = np.zeros((100,200))
for i,beta in enumerate(np.linspace(0.01,1,100)):
    print(i)
    for j,alpha in enumerate(np.linspace(0.01,2,200)):
        s = []
        rep=0
        vs=[]
        q=make_q_rand(10,20)
        while rep<50:
            try:
                vs.append(descent(alpha,beta,q,v=np.random.random(10)))
                best = vs[np.argmax([W(alpha,beta,q,v) for v in vs])]
                s.append(spec(best))
                rep+=1
            except np.linalg.LinAlgError:
                pass
        hmap_rand[i,j]=np.nanmean(s)
                

hmap_neighb = np.zeros((100,200))
for i,beta in enumerate(np.linspace(0.01,1,100)):
    print(i)
    for j,alpha in enumerate(np.linspace(0.01,2,200)):
        s = []
        rep=0
        vs=[]
        while rep<50:
            try:
                vs.append(descent(alpha,beta,make_q_neighbor(10),v=np.random.random(10)))
                best = vs[np.argmax([W(alpha,beta,make_q_neighbor(10),v) for v in vs])]
                s.append(spec(best))
                rep+=1
            except np.linalg.LinAlgError:
                pass
        hmap_neighb[i,j]=np.nanmean(s)

np.savetxt('hmap_rand.txt',hmap_rand)
np.savetxt('hmap_neighb.txt',hmap_neighb)

#%%
alphas = np.linspace(0.01,2,200)
betas = np.linspace(0.01,1,100)
fig,axes = plt.subplots(ncols=2,figsize=(20,10))
axes[0].imshow(hmap_neighb, origin='lower', cmap=plt.cm.magma)
axes[0].plot(100*alphas[:100],100*3/((4)*alphas[:100]),'cadetblue',lw=3)
axes[0].plot(100*alphas[100:],100*(6-np.sqrt(5)*alphas[100:])/(5*alphas[100:]),'cadetblue',lw=3)
axes[0].vlines(100.0,0,100,color='orange')
axes[0].set_ylim(0,100)
axes[0].set_xlim(0,200)
axes[1].imshow(hmap_rand, origin='lower', cmap=plt.cm.magma)
axes[1].vlines(100.0,0,100,color='orange')
axes[1].set_ylim(0,100)
axes[1].set_xlim(0,200)
axes[0].set_xticks(np.arange(201)[::20])
axes[0].set_xticklabels(list(np.round(alphas[::20]-.01,3))+[2.0],rotation='vertical')
axes[1].set_xticks(np.arange(201)[::20])
axes[1].set_xticklabels(list(np.round(alphas[::20]-.01,3))+[2.0],rotation='vertical')
axes[1].set_yticks(np.arange(101)[::20])
axes[1].set_yticklabels(list(np.round(betas[::20]-.01,3))+[1.0],rotation='vertical')
axes[0].set_yticks(np.arange(101)[::20])
axes[0].set_yticklabels(list(np.round(betas[::20]-.01,3))+[1.0],rotation='vertical')
