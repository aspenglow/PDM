# Geometric deep learning to study information propagation in temporal networks
Code base of master thesis in EPFL
## 1. Generate random graph 
(default: Erdo-Renyi * 500, SBM * 500)

```shell
python graph_generate.py
```

## 2. Compute final graph layout with FDL algorithm

```shell
python neulay_2.py --data-dir="graphs/erdos_renyi" --layout-dir="layouts/erdos_renyi"
python neulay_2.py --data-dir="graphs/sbm" --layout-dir="layouts/sbm"
```

## 3. Train and test model to predict final layouts given different graph structures

```shell
python main.py
```

## Notice: 
1. split dataset will be saved in graph root dir.
2. graph and layout directory structure: 
    graph_root_dir -> graph_dirs -> graphs
    layout_root_dir -> layout_dirs -> layouts