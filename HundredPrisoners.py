import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

def do_simulation(seed=None, N=100, K=100, do_plot=False, show_steps=False):
    """
    Run a simulation of the prisoner algorithm
    Parameters
    ----------
    seed: int
        Random seed to use (or none)
    N: int
        Number of prisoners
    K: int
        Rotation time to next "prisoner x"
    do_plot: boolean
        Whether to plot the matrix at the end
    show_steps: boolean
        Whether to output frames showing steps of the algorithm
    Returns
    -------
    {'d':int
        Number of days elapsed,
     'coverage': float
        Percentage of prisoners who marked each other as seen,
     'seen': ndarray(N, N)
        Array of seen; seen(i, j) = 1 if i=j or if i has seen
        j's light}
    """
    if seed:
        np.random.seed(seed)
    # Put a 1 if i=j, or if prisoner i knows that prisoner j 
    # has been in the warden's office
    seen = np.eye(N)
    # Keep track of whether each prison has visited the warden
    # (for validation at the stopping condition)
    visited_warden = np.zeros(N)
    d = 0
    light_on = False
    while not np.max(np.sum(seen, 1)) == (N-1):
        if d%K == 0 and show_steps:
            plt.clf()
            plt.imshow(seen)
            plt.title("d = %i (%.3g years)"%(d, d/365.0))
            plt.savefig("%i.png"%(int(np.floor(d/K))))
        # Loop until one prisoner is sure the N-1 others have been there
        x = int(np.mod(np.floor(d/K), N))
        with_warden = np.random.randint(N)
        visited_warden[with_warden] = 1
        if with_warden == x:
            light_on = True
        elif light_on:
            seen[with_warden, x] = 1
        if d%K == (K-1):
            light_on = False
        d += 1
    assert np.sum(visited_warden) == N
    coverage = 100*np.sum(seen)/float(seen.size)
    if do_plot:
        plt.imshow(seen)
        plt.title("%.3g %% Coverage"%(coverage))
        plt.show()
    return {'d':d, 'coverage':coverage, 'seen':seen}

def vary_parameters(NTrials = 100, Ks = list(range(50, 901, 50))):
    """
    Do a test varying the parameters
    Parameters
    ----------
    NTrials: int
        Number of trials per parameter
    Ks: list(int)
        The K parameter to try
    """
    trials = np.zeros((NTrials, len(Ks)))
    plt.figure(figsize=(12, 6))
    for i, K in enumerate(Ks):
        for t in range(NTrials):
            res = do_simulation(K=K)
            trials[t, i] = res['d']/365.0
            print("Trial %i, %i days (%.3g years), coverage=%.3g"%(t, res['d'], res['d']/365.0, res['coverage']))
        sio.savemat("res.mat", {"Ks":Ks, "trials":trials})
        plt.clf()
        sns.boxplot(data=trials)
        plt.xticks(np.arange(len(Ks)), ["%i"%k for k in Ks])
        plt.xlabel("Prisoner Rotation Days")
        plt.ylabel("Years")
        plt.title("Chris's Prison Algorithm")
        plt.savefig("Results.svg", bbox_inches='tight')

if __name__ == '__main__':
    vary_parameters()
    #print(do_simulation(K=500, show_steps=True)['d']/365)