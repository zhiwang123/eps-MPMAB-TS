import numpy as np
import module as mx
import pandas as pd
import argparse
import plot

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

        # Algorithm 5: robustTS
        algRobustTSV = mx.RobustTS(ground_truth_epsilon, problem_instance, sizeI)

        algRobustAgg.Run()
        print("RobustAgg done.")

        algBaseline.IndUCB()
        print("Ind-UCB done.")

        algTS.IndTS()
        print("Ind-TS done.")

        algRobustTS.Run()
        print("robustAgg-TS done.")

        algRobustTSV.Run_everupdating()
        print("robustAgg-TS-V done.")

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
        df = pd.concat([df, pd.DataFrame(
            {'Round': np.arange(T, dtype=int), 'Cumulative Collective Regret': algRobustTSV.CollectiveRegret(),
             'Cumulative Collective Pseudo-Regret': algRobustTSV.CollectivePseudoRegret(),
             'Algorithm': 'RobustAgg-TS-V', 'instance_num': int(i)})])

        data = pd.concat([data, df])
        # </editor-fold>

    data.to_csv('data/data_ical=%i'%int(sizeI) + '.%i'%int(T) + 'x%i'%int(num_instances) +'.csv',index=False)

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

    #compareRegretSizeI(args.num_subpar_arms, args.time_horizon, args.num_instances)
    #plot.plot(args.num_subpar_arms, args.time_horizon, args.num_instances)
    for i in range(10):
        plot.plot(i, args.time_horizon, args.num_instances)