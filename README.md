# A general geometric deep learning model for network layouts
Code base of master thesis in EPFL
## 1. Generate random graph 

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
1. Split dataset will be saved in graph root dir.
2. Graph and layout directory structure: \
    experiment_root_dir -> graph_root_dir -> graph_dirs -> graphs \
    experiment_root_dir -> layout_root_dir -> layout_dirs -> layouts
3. Examples of shell scripts: \
    ex_generalize_blocks.sh \
    ex_generalize_nodes.sh