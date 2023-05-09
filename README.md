# Geometric deep learning to study information propagation in temporal networks
Code base of master thesis in EPFL
## 1. Generate random graph 
(default: Erdo-Renyi * 500, SBM * 500)

```shell
python graph_generate.py
```

## 2. Compute final graph layout with FDL algorithm

```shell
python neulay/neulay_2.py
```

## 3. Train and test model to predict final layouts given different graph structures

```shell
python main.py
```