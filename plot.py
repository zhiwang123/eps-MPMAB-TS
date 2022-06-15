import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot(sizeI, T, num_instances):

    num_algorithms = 4

    data = pd.read_csv('data/data_ical=%i'%int(sizeI) + '.%i'%int(T) + 'x%i'%int(num_instances) +'.csv')
    a = data.to_numpy()

    # Algorithm, Pseudo-Regret, Regret, Time, instance_id
    regret_data = a[:,2]
    regret_data = regret_data.reshape((num_algorithms * num_instances, T))

    robustAggData = np.zeros((num_instances, T))
    indUCBData = np.zeros((num_instances, T))
    indTSData = np.zeros((num_instances, T))
    robustTSData = np.zeros((num_instances, T))

    # Hard-coded
    for i in range(regret_data.shape[0]): # Four algorithms
        if i % num_algorithms == 0:
            robustAggData[int(i / num_algorithms)] = regret_data[i]
        elif i % num_algorithms == 1:
            indUCBData[int(i / num_algorithms)] = regret_data[i]
        elif i % num_algorithms == 2:
            indTSData[int(i / num_algorithms)] = regret_data[i]
        elif i % num_algorithms == 3:
            robustTSData[int(i / num_algorithms)] = regret_data[i]

    # Compute means and std
    robustAggMean = np.mean(robustAggData, axis = 0)
    robustAggStd = np.std(robustAggData, axis = 0)
    indUCBMean = np.mean(indUCBData, axis = 0)
    indUCBStd = np.std(indUCBData, axis = 0)
    indTSMean = np.mean(indTSData, axis = 0)
    indTSStd = np.std(indTSData, axis = 0)
    robustTSMean = np.mean(robustTSData, axis = 0)
    robustTSStd = np.std(robustTSData, axis = 0)

    # plot
    mpl.style.use('seaborn')
    plt.figure(figsize=(9,8))

    plt.plot(np.arange(T, dtype=int), robustAggMean, label="RobustAgg(0.15)")
    plt.fill_between(np.arange(T, dtype=int), robustAggMean + robustAggStd, robustAggMean - robustAggStd, alpha=0.2)

    plt.plot(np.arange(T, dtype=int), indUCBMean, label="Ind-UCB")
    plt.fill_between(np.arange(T, dtype=int), indUCBMean + indUCBStd, indUCBMean - indUCBStd, alpha=0.2)

    plt.plot(np.arange(T, dtype=int), indTSMean, color="peru", label="Ind-TS")
    plt.fill_between(np.arange(T, dtype=int), indTSMean + indTSStd, indTSMean - indTSStd, alpha=0.2, color="peru")

    plt.plot(np.arange(T, dtype=int), robustTSMean, label="RobustAgg-TS(0.15)")
    plt.fill_between(np.arange(T, dtype=int), robustTSMean + robustTSStd, robustTSMean - robustTSStd, alpha=0.2)

    plt.xlabel("Round", fontsize=20, horizontalalignment='center')
    plt.ylabel("Cumulative Collective Regret", fontsize=20)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.xlim(-100, T+100)
    plt.ylim(-100, 30000)
    plt.legend(loc="upper left", fontsize=19, frameon=True)

    plt.savefig("plots/ical=%i"%sizeI + "_%i"%T + "x%i"%num_instances + ".png", dpi=300, bbox_inches='tight')