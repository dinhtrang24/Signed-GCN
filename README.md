# Signed-GCN

This is the source code for the manuscript "On the modelling and impact of negative edges in graph convolutional networks for node classification".

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Reproduce results for synthetic data
Create folder to save results
```
mkdir job_id{1..25}
```
We have 25 files of results which correspond to 25 combinations of within and between degree for each scenario. We generate between and within-community edges varying in ${1, 3, 6, 12, 24}$ . For example, job_id1 corresponds to the case when within degree = 1 and between degree =1; $\dots$, job_id25 corresponds to the case when within degree = 24 and between degree =24.

If we want to obtain the result for the case when within degree = 1 and between degree =1, we implement:

```
python synthetic_data.py 1 1 0.1 0.3 0 0 1 1 1 1 1
```
The above eleven parameters correspond to:
- unique number of random seeds (here we have 30 random seeds in total corresponding to 30 splits)
- number of runs in each split
- training set percentage
- validation set percentage (the remaining is tet set)
- negative link noise percentage
- positive link noise percentage
- within degree of Group 1
- within degree of Group 2
- within degree of Group 3
- between degree of the whole graph
- job id

## Reproduce results for real networks
Create folder to save results
```
mkdir result
```
For Cora data, if we want to obtain the result for the case when there is no edge noise and no feature noise, we run this command:
```
python real_network.py 1 1 0 0 0
```
The above five parameters correspond to:
- unique number of random seeds corresponding to different weight initializations
- number of runs
- negative link noise percentage
- positive link noise percentage
- feature noise



