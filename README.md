# MNIST-NET10
This complex heterogenous fusion is composed of two heterogenous ensembles FS1 and FS2:

#FS1 (CapsNet|MCDNN|DropConnect_2|CapsNet|MCDNN| DropConnect_1|DropConnect_2|Network3|Dropconnect_2) can be built using these codes:
1. Pre-trained CapsNet downloaded from: 

https://github.com/Sarasra/models/tree/master/research/capsules

2. MCDNN network was obtained from:

https://github.com/xanwerneck/ml_mnist

3. Network3 with data augmentation (see Network3.py)

4. DropConnect with data augmentation (see DropConnect.py)


#FS2 (ECOC|PrE|MLP→LS|MLP) can be built using these codes:

1. CapsNet as data transformer from: 

https://github.com/Sarasra/models/tree/master/research/capsules

2. The needed codes (in Matlab) are available from: 

You can download the paper from this link: https://www.researchgate.net/publication/340954880_MNIST-NET10_A_heterogeneous_deep_networks_fusion_based_on_the_degree_of_certainty_to_reach_01_error_rate_Ensembles_overview_and_proposal
 http://www.tsc.uc3m.es/~ralvear/Software.htm  

