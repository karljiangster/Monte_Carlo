
# coding: utf-8

# # 37 - Gibbs Sampling 
# 
# The Ising Model is given by $\pi(\sigma) = \textrm{exp}(\sum_{\overrightarrow{i} \textrm{~} \overrightarrow{j}} \sigma_{\overrightarrow{i}} \sigma_{\overrightarrow{j}}) / Z$, where $Z$ is the normalization constant (sum of the denisities across the state space of size $2^{L^2}$, where $L$ is the length of the lattice) 
# <br> 
# 
# If not specificed, we'll be using n = 1 million samples, $\beta = 0.2$, and lattice size 4 (4x4 lattice) 

# In[507]:


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import acor
import numpy as np 

BOLT = 100 #1.3807 * 10**(-23) #actually this numbers get rather large / small 


# In[508]:


def initialize_lattice(l):
    space = [-1, 1]
    L = [] 
    for i in range(l):
        L.append( np.random.choice(space, size = l) ) 
    return( np.matrix(L) )     
    

def get_neighbors(L, i, j):
    '''
    params: 
        - L: nxn np matrix representing the periodic lattice 
        - i: int, x coordinate on lattice L 
        - j: int, y coordinate on lattice L 
    returns: 
        length 4 tuple of coordinates (x,y) of the neighbor
        (NORTH, SOUTH, EAST, WEST)
    note: using np matrix coordinates so i -> row index, j -> columns index 
    '''
    l = L.shape[0]
    if j == 0: 
        WEST = (i, l - 1)
    else: 
        WEST = (i, j - 1)
    if j == (l-1):
        EAST = (i, 0)
    else: 
        EAST = (i, j + 1,)
    
    if i == 0: 
        NORTH = (l - 1, j)
    else: 
        NORTH = (i - 1, j)
    if i == (l-1):
        SOUTH = (0, j)
    else: 
        SOUTH = (i + 1, j)
    
    return (NORTH, SOUTH, EAST, WEST)
    
def ising_conditional(L, beta, i, j): 
    '''
    params: 
        - L: nxn np matrix representing the periodic lattice 
        - i: int, x coordinate on lattice L 
        - j: int, y coordinate on lattice L 
        - beta: float, 1 / (boltzman_constant * kelvin)
    returns: 
        - conditional probabillity p(sigma_{ik} = 1| sigma_{others} ) = 1 - p(sigma_{ik} = -1| sigma_{others} )
    '''
    neighbors = get_neighbors(L, i, j)
    mag_neighbor = 0
    #print(neighbors)
    for point in neighbors: 
        #print(point)
        sigma = L[point[0], point[1]]
        #print(sigma)
        mag_neighbor += sigma
    #print(mag_neighbor)
    power = 2 * mag_neighbor * beta
    pos = np.e**(power)
    neg = np.e**(-power)
    
    return pos / (pos + neg)


# In[509]:


#Test neighbors 
lattice = initialize_lattice(3)
points = [(0,0), (1,1), (2,2), (1,0)]
neighbors = [   [(2, 0), (1, 0), (0, 1), (0, 2)],
                [(0, 1), (2, 1), (1, 2), (1, 0)],
                [(1, 2), (0, 2), (2, 0), (2, 1)],
                [(0, 0), (2, 0), (1, 1), (1, 2)]
            ]
for index in range(len(points)):
    expect = neighbors[index]
    x = points[index][0]
    y = points[index][1]
    actual = get_neighbors(lattice, x, y) 
    print("expected: {}".format(expect))
    print("actual: {}\n".format(actual))


# In[510]:


#Test conditional, just a quick visual inspection will do.  
beta = 0.2 
print(lattice) 

print( ising_conditional(lattice, beta, 0, 0 ) ) 
print( ising_conditional(lattice, beta, 1, 0 ) )
print( ising_conditional(lattice, beta, 1, 2) ) 


# In[511]:


def gibbs_ising_magnetization_deterministic(L, beta, n):
    '''
    Gibbs sampling resampling each point on the lattice iteratively.  
    params: 
        - n: number of samples 
        - L: nxn np matrix representing the periodic lattice. Values should ONLY be {-1, 1}
    returns: 
        - numpy vector of length n where each is the magnetization of the lattice
    '''
    space = [1, -1]
    l = L.shape[0]
    mags = [] 
    i = 0 
    j = 0 
    for k in range(n): 
        if j == l: 
            j = 0 
            i += 1 
        if i == l: 
            i = 0 
        #print(i, j)
        p = ising_conditional(L, beta, i, j)
        sigma_ik = np.random.choice(space, p = [p, 1-p])
        L[i, j] =  sigma_ik#now update the lattice
        mag = np.sum(L)
        mags.append(mag)
        j += 1 

    return mags 


# Let's test out the gibbs sampler for magnetization for on a 4x4 by lattice. First let's do deterministically by traversing the lattice. 

# In[301]:


#Test gibbs sampling 
lattice = initialize_lattice(l=4)
n = 10**7
gibbs_4_d = gibbs_ising_magnetization_deterministic(lattice, beta, n=n)


# In[302]:


plt.hist(gibbs_4_d, normed = 1)
plt.title('Histogram: 4x4 Magnetization - Deterministic')
plt.xlabel('Magnetization')
plt.ylabel('Probability')
plt.show()


# In[320]:


tau, mean, sigma = acor.acor(gibbs_4_d )
print("The gibbs sampling estimator (deterministic) is {}, with integrated autocorrelation time of {}".format(mean, tau))


# Now let's try gibbs sampling on the ising model with randomly selected points on the same 4x4 lattice: 

# In[309]:


def gibbs_ising_magnetization_random(L, beta, n): 
    '''
    Gibbs sampling using a random point on the lattice to resample. 
    params: 
        - n: number of samples 
        - L: nxn np matrix representing the periodic lattice. Values should ONLY be {-1, 1}
    returns: 
        - numpy vector of length n where each is the magnetization of the lattice
    '''
    space = [1, -1]
    l = L.shape[0]
    indices = list(range(l))
    mags = [] 
    
    x = np.random.choice(indices, size = n)
    y = np.random.choice(indices, size = n)
    for i, j in zip(x, y): 
        #print(i, j)
        p = ising_conditional(L, beta, i, j)
        sigma_ik = np.random.choice(space, p = [p, 1-p])
        L[i, j] =  sigma_ik #now update the lattice
        mag = np.sum(L)
        mags.append(mag)
    return mags 
    


# In[312]:


gibbs_4_r = gibbs_ising_magnetization_random(lattice, beta, n=n)


# In[313]:


plt.hist(gibbs_4_r, normed = 1)
plt.title('Histogram: 4x4 Magnetization - Random')
plt.xlabel('Magnetization')
plt.ylabel('Probability')
plt.show()


# In[321]:


tau, mean, sigma = acor.acor(gibbs_4_r )
print("The gibbs sampling estimator (random) is {}, with integrated autocorrelation time of {}".format(mean, tau))


# The autocorrelation for the random case (without replacement) is noticably higher than for the deterministic case. This is mainy because it is likely that we resample the same points (within near time steps) more often, leading to a slower change in the Ising model. Thus, we prefer to determinstically sweep through the lattice, which will be faster   

# ## Changing the size of the Lattice 
# 
# For the same number of samples and temperature, we observe what happens when the lattice size changes. Not going to run this part because temperature took almost an hour and it's getting kind of late. 

# In[ ]:


sizes = list(range(3, 30, 5))
acor_data = [] 
for l in sizes: 
    lattice_l = initialize_lattice(l=l)
    gibbs_l_d = gibbs_ising_magnetization_deterministic(lattice_l, beta, n=10**7)
    tau_l, mean_l, sigma_l = acor.acor(gibbs_l_d )
    acor_data.append([tau_l, mean_l, sigma_l])
    print("The gibbs sampler estimator for lattice size {} is {}, with integrated autocorrelation time of {}".format(l, mean_l, tau_l))


# Even though it may not have run, we expact that as the lattice size increases, so does the integrated autocorrelation time. This makes since at each step we are unlikely to make significant changes to the lattice, hence the magnetization for each step is most likely going to be the same. 

# ## Changing the Temperature 
# 
# For the same number of samples and lattice size, we observe what happens when the temperature changes:  

# In[487]:


powers = np.linspace(start = -2, stop = -0.2, num = 5)
acor_beta = [] 
for p in powers:
    b = 10**p
    gibbs_b_d = gibbs_ising_magnetization_deterministic(lattice, b, n=10**7)
    tau_b, mean_b, sigma_b = acor.acor(gibbs_b_d)
    acor_beta.append([tau_b, mean_b, sigma_b])
    print("The gibbs sampler for beta = {} is {}, with integrated autocorrelation time of {}".format(b, mean_b, tau_b))


# We observe that as the beta increases (temperature decreases), the integrated autocorrelation increases as well. High temperatures means a higher probabillity of a point to take on the charge of its neighbors, so once we reach a magnetization of $L^2$ or $-L^2$, changes in magnetization will be rare.

# # Metropolis Hastings, Ising Magnetization) 

# In[276]:


def metropolis_hasting_ising(n, L, beta): 
    '''
    Uses Metropolis Hastings to sample the ising model magentization
    params:  
        - n: number of samples to obtain  
        - L: nxn np matrix representing the periodic lattice. Values should ONLY be {-1, 1}
        - beta: unsigned int, represents the temperature. Higher beta means lower temperature. 
    returns: 
        - numpy vector of length n where each is the magnetization of the lattice
    '''
    space = [1, -1]
    l = L.shape[0]
    indices = list(range(l))

    mags = []
    x = np.random.choice(indices, size = n)
    y = np.random.choice(indices, size = n)
    for i, j in zip(x, y): 
        p = ising_conditional(L, beta, i, j)
        y_ik = np.random.choice(space, p = [p, 1-p])
        
        ratio = step_density_ratio(L, beta, i, j)
        p_acc = min([1, ratio])
        accept = np.random.choice([True, False], p = [p_acc, 1 - p_acc])
        if accept:
            y_ik = np.random.choice(space, p = [p, 1 - p])
        L[i, j] =  y_ik #now update the lattice
        mag = np.sum(L)
        mags.append(mag) 

    return mags 

def step_density_ratio(L, beta, i, j): 
    '''
    returns: 
        the ratio of pi(y) / pi(x) 
    '''
    neighbors = get_neighbors(i=i, j=j, L=L)
    mag_neighbors = 0 
    for point in neighbors: 
        mag_neighbors += L[point[0], point[1]]
    #print(mag_neighbors)
    power = -4 * beta * L[i,j] * mag_neighbors
    #print(power)
    return np.exp(power) 


# In[317]:


mh_4 = metropolis_hasting_ising(n, lattice, beta)


# In[318]:


plt.hist(mh_4, normed = 1)                                                     
plt.title('Histogram: 4x4 Magnetization - Random')
plt.xlabel('Magnetization')
plt.ylabel('Probability')
plt.show()


# In[514]:


tau_mh, mean_mh, sigma_mh = acor.acor(mh_4)   
print("The MCMC estimator (Metroplis Hasting) is {}, with integrated autocorrelation time of {}".
      format(mean_mh, tau_mh))


# The integrated autocorrelation time for MH is about twice as large as the one from the determinisitc MH scheme. This is most likely due to having the same sample when we "reject" the next step and our Ising model stays the same (Thus leading the lattice to be more similar to previous iterations).

# # Jarzynski’s method 
# 
# ## Without Resampling 
# We need a scheme that can preserve $\pi_k$, so gibbs sampling works perfectly for this. That is, for each trajectory j at step k $X^{(k,j)}$ we will randomly select a point on the lattice and resample. Below are some functions to help us out (specifically to update the weights and lattice of each trajectory: 

# In[513]:


def update_weights_ising(L, prev_weight, n, beta):
    '''
    updates the weights: W_k = W_{k-1} * pi^{1 / N-1} 
    params: 
        - L: lxl np matrix the lattice 
        - prev_weight: float, W_{k-1} of trajectory j 
        - n: int, total number of steps
        - beta: float
    returns: 
        new weight 
    '''
    #let's calc w_k = pi(x_k)^(1/(N - 1) ) 
    l = L.shape[0]
    inter_sum = 0 
    for i in range(l): 
        for j in range(l): 
            neighbors = get_neighbors(i=i, j=j, L=L)
            for p in neighbors: 
                inter_sum += L[i, j] * L[p[0], p[1]]  
    pi_xt = np.exp(beta * inter_sum)
    pwr = 1/(n-1)
    w_k = pi_xt**(pwr)
    W_k = prev_weight * w_k 
    return(W_k)

##quick test
test_L = np.matrix([[1,1],[1,1]])
traj_len = 10
prev_weight = 1/traj_len 
#update_weights_ising(test_L, prev_weight, traj_len, beta) #yay we good 


# In[512]:


def update_gibbs_lattice_list(lattices, weights, l, beta, n): 
    '''
    Updates each trajectory and the trajectory's weights 
    params: 
        l: dimension of lattice 
        lattices: length m list of lxl lattices 
        weights: current weight vector for step k 
        beta: float, proportional to inverse of temperature (Kelvin)
        n: int, number of steps to reach pi  
    returns: 
        (length m updated lattice list, length m array-like of weights)  
    '''
    m = len(lattices)
    indices = list(range(l))
    x_coords = np.random.choice(indices, size = m)
    y_coords = np.random.choice(indices, size = m)
    space = [1, -1]
    for t_i in range(m): #t_i = trajectory_i  
        L = lattices[t_i]
        x = x_coords[t_i]; y = y_coords[t_i] #chosen point on lattice to resample  
        
        p = ising_conditional(L, beta, x, y)
        sigma_ik = np.random.choice(space, p = [p, 1-p])
        L[x, y] =  sigma_ik #now update the lattice
        
        lattices[t_i] = L #update the lattice list with the updated lattice 
        weights[t_i] = update_weights_ising(beta=beta, L=L, n=n, prev_weight=weights[t_i])
    weights = weights / np.sum(weights) # normalize weights to sum to 1 
    return lattices, weights
    
#Quick test
l1 = np.matrix([[1,1], [1,1]])
l2 = -1 * np.matrix([[1,1], [1,1]])
l3 = np.matrix([[1,-1], [1,-1]])
lattices_test = [l1, l3]
print(lattices_test)
test_weights = [0.5, 0.5]
update_gibbs_lattice_list(lattices_test, test_weights, l = 2, beta = beta, n = traj_len)


# More functions, for resampling the weights. 

# In[469]:


def generate_lattices(M, l):
    '''
    generates M lxl lattices 
    returns: 
        len M list of lxl np matrices 
    '''
    lattices = [] 
    for i in range(M):
        lattices.append(initialize_lattice(l))
    return lattices

def resample_bernoulli(weights, l): 
    '''
    Generates new copy numbers N_k using Bernoilli Resampling technique 
    params: 
        weights - np array
        l - number of draws 
    returns: 
         np.matrix n x l where each index [i,j] contains copy number sample i of sample x_j with weight w_j 
    '''
    n = len(weights)
    NW = n * weights
    NW_floor = NW.astype(int)
    
    resamples = [] 
    for i in range(l):
        u = np.random.uniform(size = n)
        cond = (u <  NW - NW_floor)
        resamples.append( NW_floor + cond ) 
    return np.matrix( resamples ) 

def bernoulli_lattice_resample(lattice_list, weights):
    '''
    Uses bernoulli resampling to get a new list of trajectories 
    '''
    lattice_list_new = [] 
    copy_numbers = resample_bernoulli(np.array(weights), 1)
    for i in range(len(weights)):
        cn = copy_numbers[0, i]
        for j in range(cn): 
            lattice_list_new.append(lattice_list[i])
    return lattice_list_new

#quick test for bernoulli resampling 
l1 = np.matrix([[1,1], [1,1]])
l2 = -1 * np.matrix([[1,1], [1,1]])
l3 = np.matrix([[1,-1], [1,-1]])
lattices_test = [l1, l2, l3]
weights = [0.1, 0.3, 0.6]
bernoulli_lattice_resample(lattices_test, weights)


# In[496]:


def generate_lattices(M, l):
    '''
    generates M lxl lattices 
    returns: 
        len M list of lxl np matrices 
    '''
    lattices = [] 
    for i in range(M):
        lattices.append(initialize_lattice(l))
    return lattices


def jarzynski_ising(M, n, l, beta = 0.2, resample = False): 
    '''
    The main function for jarzynski_ising. 
    Using density update pi_k = pi^(k/n-1) so we can start we a uniformally {-1,1} sampled lattice 
    params: 
        M - int, number of trajectories 
        n - int, number of steps to reach pi 
        l - int, dimension of lattice 
        resample - If True, resamples the weights at the beginning of each update step. 
    returns: 
        length M chain of magnetizations 
    '''
    lattices = generate_lattices(M, l)
    weights = [1/M] * M
    for k in range(n):
        if resample: 
            lattices = bernoulli_lattice_resample(lattices, weights)
            M_n = len(lattices)
            weights = [1/M_n] * M_n
        #update each lattice, weights with gibbs 
        lattices, weights = update_gibbs_lattice_list(lattices, weights, l, beta, n)
    return lattices, weights


def compute_weighted_mag(lattice_list, weights): 
    '''
    computes the weighted average of the lattices 
    '''
    avg = 0 
    for w, L in zip(weights, lattice_list): 
        avg += w * np.sum(L)
    return avg 

#quick test 
lattice_list, weights = jarzynski_ising(M=2, n = 2, l = 2)
compute_weighted_mag(lattice_list, weights)


# To evaluate the performance, let us observe how quickly Jarzynki converges to 0 as we increase the number of trajectories $M$ For $N = 50$ steps on the $\beta = 0.2$ 4x4 lattice:  

# In[ ]:


def get_trend(lattice_list, weights):
    '''
    returns: 
        a list of estimators of the expected magnetization using on estimators 1:k 
    '''
    est = [] 
    for k in range(len(lattice_list)): 
        l_tmp = lattice_list[0:k]
        w_tmp = weights[0:k]
        w_tmp = w_tmp / np.sum(w_tmp)
        est.append(compute_weighted_mag(l_tmp, w_tmp))
    return est 


lattice_list, weights = jarzynski_ising(M=1000, n = 50, l = 4)
trend = get_trend(lattice_list, weights)
traj_count = list(range(len(lattice_list)))


# In[501]:


plt.plot(traj_count, trend)
plt.xlabel("Number of Trajectories")
plt.ylabel("Est. Expected Magnetization")
plt.show()


# At about 900 trajectories, we our estimator seems to hit a stable form at 0. For computation this corresponds to 900 trajectories x 50 steps, which is far less than gibbs or MH for the same ising model. (Though we do have to traverse the lattice for each update).

# ### Increasing the number of steps
# 
# Let's see what happens for different sizes of n: 

# In[475]:


M = 100
l = 4

n_list = list(range(5, 400, 25))
means = [] 
weight_vars = [] 
for N in n_list:  
    lattice_list, weights = jarzynski_ising(M = M, n = N, l = l)
    weight_vars.append( np.var(weights) ) 
    means.append( compute_weighted_mag(lattice_list, weights) ) 


# In[476]:


plt.plot(n_list, weight_vars)
plt.xlabel('n, the number time steps')
plt.ylabel('variance of the weight vector')
plt.title('Variance of weights as we crank up n')
plt.show()


# Based off of the variance plot, it seems that the variance reaches a limiting value at about $n = 225$. This should be the point where the majority of the points in the lattice have a magnetization of $L^2$ or $-L^2$ - the most probably case (And hence stable weights). Keeping "computational effort" in mind: as we increase n, we have to resample M lattices, so if the decrease in variance is insignificant for an increase in n, we should stop. 

# In[504]:


plt.plot(n_list, means)
plt.xlabel('n, the number time steps')
plt.ylabel('estimator of expected magnetization')
plt.title('estimator as we crank up n')
plt.show()


# Based off the estimator plot, we see that we increase the number of steps, the more accurate our estimator is (seems to have limit value at 0). The makes sense because again, all trajectories will approach all positive or negative magnetization. 
# 
# 

# ## With Resampling 
# 
# Now let's try with Bernoulli Resampling of the Weights: 

# In[481]:


n_list = list(range(5, 50, 5))
means_bernoulli = [] 
weight_vars_bernoulli = [] 
for N in n_list:  
    lattice_list, weights = jarzynski_ising(M = M, n = N, l = l, resample=True)
    weight_vars_bernoulli.append( np.var(weights) ) 
    means_bernoulli.append( compute_weighted_mag(lattice_list, weights) ) 


# In[482]:


plt.plot(n_list, weight_vars_bernoulli)
plt.xlabel('n, the number time steps')
plt.ylabel('variance of the weight vector')
plt.title('Variance of weights (using resampling) as we crank up n')
plt.show()


# The variance of the weight vector is small when we resample the trajectories with respect to their weights, naturally because we have a new vector of weights of weight of $\frac{1}{M_k}$, where most of the copy numbers will fall on the latices with $L^2$ or $-L^2$ magnetization. Notice that compared to without resampling, the variance is not only smaller, but also approaches its limiting value significnantly faster (at about $n = 15$ the variance doesn't decrease significantly). 

# ## Comparison 
# 
# The main edge that sequential importance sampling has over gibbs sampling and MH is that we can start at some initial distribution. We can't simply sample from $\pi(\sigma)$, so we started with a uniformally sampled lattice. This is problematic when we do gibbs and MH. Jarzynski's Method allows us to start at a initial state that is not $\pi$ that will converge to an unbiased estimator (if we know the multiplicative constant) for $\pi f$. 
# 
# <br> 
# However, one problem with Jarzinsky is computation time. While we don't need as large $N$ (number of time steps, only 15 when using resampling compared to 1 million for a 4x4 lattice for a reasonable estimator from MH and gibbs), but we need to a good number of trajectories $M$, resulting in O(MN) updates. We also need to continually compute the interaction for the density of the lattice at each weight update, so we need to traverse through the entire lattice for each trajectory. So as we increase the size of the lattice, computation will become even more expensive. 
# 
