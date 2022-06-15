import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot(sizeI, T, num_instances):

    num_algorithms = 4

    data = pd.read_csv('data/percentages_ical=%i' % int(sizeI) + '.%i' % int(T) + 'x%i' % int(num_instances) + '.csv')
    a = data.to_numpy()

    # instance_num, algorithm, opt, near-opt, subpar
    opt_data = a[:, 2]
    nearopt_data = a[:, 3]
    subpar_data = a[:, 4]

    robustagg_data = np.zeros((3, num_instances))
    inducb_data = np.zeros((3, num_instances))
    indts_data = np.zeros((3, num_instances))
    robustts_data = np.zeros((3, num_instances))

    # Hard-coded
    for i in range(opt_data.shape[0]):
        if i % num_algorithms == 0:
            robustagg_data[0][int(i / num_algorithms)] = opt_data[i]
            robustagg_data[1][int(i / num_algorithms)] = nearopt_data[i]
            robustagg_data[2][int(i / num_algorithms)] = subpar_data[i]
        elif i % num_algorithms == 1:
            inducb_data[0][int(i / num_algorithms)] = opt_data[i]
            inducb_data[1][int(i / num_algorithms)] = nearopt_data[i]
            inducb_data[2][int(i / num_algorithms)] = subpar_data[i]
        elif i % num_algorithms == 2:
            indts_data[0][int(i / num_algorithms)] = opt_data[i]
            indts_data[1][int(i / num_algorithms)] = nearopt_data[i]
            indts_data[2][int(i / num_algorithms)] = subpar_data[i]
        elif i % num_algorithms == 3:
            robustts_data[0][int(i / num_algorithms)] = opt_data[i]
            robustts_data[1][int(i / num_algorithms)] = nearopt_data[i]
            robustts_data[2][int(i / num_algorithms)] = subpar_data[i]

    robustagg_mean = np.zeros(3)
    robustagg_std = np.zeros(3)
    inducb_mean = np.zeros(3)
    inducb_std = np.zeros(3)
    indts_mean = np.zeros(3)
    indts_std = np.zeros(3)
    robustts_mean = np.zeros(3)
    robustts_std = np.zeros(3)

    for i in range(3):
        robustagg_mean[i] = np.mean(robustagg_data[i], axis=0)
        robustagg_std[i] = np.std(robustagg_data[i], axis=0)
        inducb_mean[i] = np.mean(inducb_data[i], axis=0)
        inducb_std[i] = np.std(inducb_data[i], axis=0)
        indts_mean[i] = np.mean(indts_data[i], axis=0)
        indts_std[i] = np.std(indts_data[i], axis=0)
        robustts_mean[i] = np.mean(robustts_data[i], axis=0)
        robustts_std[i] = np.std(robustts_data[i], axis=0)

    mpl.style.use('seaborn-deep')
    plt.figure(figsize=(9, 8))

    plt.bar(np.arange(3)-0.18, robustagg_mean, yerr=robustagg_std, width=0.12, align='center',
            alpha=0.8, label='RobustAgg(0.15)')
    plt.bar(np.arange(3)-0.06, inducb_mean, yerr=inducb_std, width=0.12, align='center',
            alpha=0.8, label='Ind-UCB')
    plt.bar(np.arange(3)+0.06, indts_mean, yerr=indts_std, width=0.12, align='center',
            alpha=0.8, label='Ind-TS', color='peru')
    plt.bar(np.arange(3)+0.18, robustts_mean, yerr=robustts_std, width=0.12, align='center',
            alpha=0.8, label='RobustAgg-TS(0.15)')

    plt.xlabel("Arm Optimality", fontsize=20)
    plt.ylabel("Percentage of total pulls", fontsize=20)
    plt.xticks([0,1,2], ['Optimal', 'Near optimal', 'Subpar'], fontsize=16)
    plt.yticks([0,0.2,0.4,0.6,0.8,1], ['0', '20%', '40%', '60%', '80%', '100%'], fontsize=16)
    plt.legend(loc="upper right", fontsize=19, frameon=True)

    plt.savefig("plots/perc_ical=%i" % sizeI + "_%i" % T + "x%i" % num_instances + ".png", dpi=300,
                bbox_inches='tight')
