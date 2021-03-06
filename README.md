### Thompson Sampling for Robust Transfer in Multi-Task Bandits

To appear in ICML-2022.

Authors: Zhi Wang, Chicheng Zhang and Kamalika Chaudhuri.

The implementation in this repository is based on https://github.com/zhiwang123/eps-MPMAB (AISTATS-2021 paper titled 
"Multitask Bandit Learning Through Heterogeneous Feedback Aggregation" by Zhi Wang*, Chicheng Zhang*, Manish Kumar Singh, 
Laurel D. Riek, and Kamalika Chaudhuri.)

###### Required packages:
- numpy
- matplotlib
- pandas

###### Remarks:
This repository can be used to reproduce the empirical evaluation in the ICML-2022 paper.
To reproduce the results in Section E.1 of the paper, please switch to the branch `ts-v`.
A description of the experimental setup can be found in the paper with further details included in the AISTATS-2021 paper.
Please first create two folders `data/` and `plots/`, 
to which data and plots will be saved, respectively.

###### Usage/Examples:
- To run the experiment with `30` generated Bernoulli 0.15-MPMAB problem instances, 
 each of which has `8` subpar arms out of `10` arms and a horizon of `50000` rounds:
`python main.py --time_horizon 50000 --num_subpar_arms 8 --num_instances 30`.
