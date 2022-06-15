import numpy as np
import module as mx
import pandas as pd
import argparse
import plot
import plot_perc
import plot_regret_perc

def compareRegretSizeI(sizeI, T, num_instances):
    K = 10
    M = 20
    ground_truth_epsilon = 0.15

    data = pd.DataFrame()  # regret
    data2 = pd.DataFrame()  # arm pull percentage
    data3 = pd.DataFrame()  # regret distribution on arms

    for i in range(num_instances):
        problem_instance = mx.MPMAB(a_num_players=M, a_epsilon=ground_truth_epsilon, a_arm_count=K, a_time_horizon=T,
                                 a_sizeI=sizeI, assumption=True)

        print("instance: ", i)

        # Algorithm 1: robust aggregation (UCB)
        algRobustAgg = mx.RobustAgg(ground_truth_epsilon, problem_instance, sizeI)

        # Algorithm 2: Individual-UCB baseline
        algBaseline = mx.RobustAgg(ground_truth_epsilon, problem_instance, sizeI)

        # Algorithm 3: individual-TS baseline
        algTS = mx.RobustAgg(ground_truth_epsilon, problem_instance, sizeI)

        # Algorithm 4: robust aggregation (TS)
        algRobustTS = mx.RobustTS(ground_truth_epsilon, problem_instance, sizeI)

        algRobustAgg.Run()
        print("RobustAgg done.")

        algBaseline.IndUCB()
        print("Ind-UCB done.")

        algTS.IndTS()
        print("Ind-TS done.")

        algRobustTS.Run()
        print("robustAgg-TS done.")

        # <editor-fold desc="Concatenate data">
        df = pd.DataFrame({'Round':np.arange(T, dtype=int),
                           'Cumulative Collective Regret': algRobustAgg.CollectiveRegret(),
                           'Cumulative Collective Pseudo-Regret': algRobustAgg.CollectivePseudoRegret(),
                           'Algorithm':'RobustAgg', 'instance_num': int(i)})
        df = pd.concat([df, pd.DataFrame(
            {'Round': np.arange(T, dtype=int), 'Cumulative Collective Regret': algBaseline.CollectiveRegret(),
             'Cumulative Collective Pseudo-Regret': algBaseline.CollectivePseudoRegret(),
             'Algorithm': 'Ind-UCB', 'instance_num': int(i)})])
        df = pd.concat([df, pd.DataFrame(
            {'Round': np.arange(T, dtype=int), 'Cumulative Collective Regret': algTS.CollectiveRegret(),
             'Cumulative Collective Pseudo-Regret': algTS.CollectivePseudoRegret(),
             'Algorithm': 'Ind-TS', 'instance_num': int(i)})])
        df = pd.concat([df, pd.DataFrame(
            {'Round': np.arange(T, dtype=int), 'Cumulative Collective Regret': algRobustTS.CollectiveRegret(),
             'Cumulative Collective Pseudo-Regret': algRobustTS.CollectivePseudoRegret(),
             'Algorithm': 'RobustAgg-TS', 'instance_num': int(i)})])


        df2 = pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'RobustAgg',
                            'Opt Percentage': algRobustAgg.opt_pulls,
                            'Near-optimal Percentage': algRobustAgg.near_opt_pulls,
                            'Subpar Percentage': algRobustAgg.subpar_pulls})
        df2 = pd.concat([df2, pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'Ind-UCB',
                                           'Opt Percentage': algBaseline.opt_pulls,
                                           'Near-optimal Percentage': algBaseline.near_opt_pulls,
                                           'Subpar Percentage': algBaseline.subpar_pulls})])
        df2 = pd.concat([df2, pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'Ind-TS',
                                           'Opt Percentage': algTS.opt_pulls,
                                           'Near-optimal Percentage': algTS.near_opt_pulls,
                                           'Subpar Percentage': algTS.subpar_pulls})])
        df2 = pd.concat([df2, pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'RobustAgg-TS',
                                           'Opt Percentage': algRobustTS.opt_pulls,
                                           'Near-optimal Percentage': algRobustTS.near_opt_pulls,
                                           'Subpar Percentage': algRobustTS.subpar_pulls})])


        df3 = pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'RobustAgg',
                            'Near-optimal Regret': algRobustAgg.near_opt_regret,
                            'Subpar Regret': algRobustAgg.subpar_regret})
        df3 = pd.concat([df3, pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'Ind-UCB',
                                           'Near-optimal Regret': algBaseline.near_opt_regret,
                                           'Subpar Regret': algBaseline.subpar_regret})])
        df3 = pd.concat([df3, pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'Ind-TS',
                                           'Near-optimal Regret': algTS.near_opt_regret,
                                           'Subpar Regret': algTS.subpar_regret})])
        df3 = pd.concat([df3, pd.DataFrame({'instance_num': [int(i)], 'Algorithm': 'RobustAgg-TS',
                                           'Near-optimal Regret': algRobustTS.near_opt_regret,
                                           'Subpar Regret': algRobustTS.subpar_regret})])
        data = pd.concat([data, df])
        data2 = pd.concat([data2, df2])
        data3 = pd.concat([data3, df3])
        # </editor-fold>

    data.to_csv('data/data_ical=%i'%int(sizeI) + '.%i'%int(T) + 'x%i'%int(num_instances) +'.csv',index=False)
    data2.to_csv('data/percentages_ical=%i'%int(sizeI) + '.%i'%int(T) + 'x%i'%int(num_instances) +'.csv',index=False)
    data3.to_csv('data/regret_percentages_ical=%i'%int(sizeI) + '.%i'%int(T) + 'x%i'%int(num_instances) +'.csv',
                 index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--time_horizon', help='the time horizon T (default: 50000)',
                        type=int, default=50000, required=True)
    parser.add_argument('-I', '--num_subpar_arms', help='the number of subpar arms (an integer between 0 and K-1,'
                                                           'default: 0)',
                        type=int, default = 0, required=True)
    parser.add_argument('-R', '--num_instances', help='generate multiple problem instances and compute the average'
                                                         'performance over the instances (default: 10)',
                        type=int, default=10, required=True)
    args = parser.parse_args()

    compareRegretSizeI(args.num_subpar_arms, args.time_horizon, args.num_instances)
    plot.plot(args.num_subpar_arms, args.time_horizon, args.num_instances)
    plot_perc.plot(args.num_subpar_arms, args.time_horizon, args.num_instances)
    plot_regret_perc.plot(args.num_subpar_arms, args.time_horizon, args.num_instances)