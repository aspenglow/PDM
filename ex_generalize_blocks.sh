#!/bin/bash
#SBATCH --job-name danyang_ex_2_blocks # Name for your job
#SBATCH --nodes 1             # do not change (unless you do MPI) 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2      # reserve 4 cpus on a single node
#SBATCH --time 1200             # Runtime in minutes.
#SBATCH --mem 1000             # Reserve 10 GB RAM for the job
#SBATCH --partition gpu         # Partition to submit ('gpu' or 'cpu')
#SBATCH --qos students             # QOS ('staff' or 'students')
#SBATCH --mail-user danyang.wang@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --gres gpu:gtx1080:1            # Reserve 1 GPU for usage, can be 'teslak40', 'gtx1080', or 'titanrtx'
#SBATCH --chdir /nfs_home/dawang/PDM

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate pdm

ex_root_dir="ex_layout_2_dims/ex_generalize_blocks/nodes_300_500/"
ps_encoding_dim=30
epochs=100

# GENERATE GRAPHS
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_2_4_blocks/sbm_2_blocks" --num-sbm 200 --num-nodes-range 300 500 --num-sbm-blocks 2
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_2_4_blocks/sbm_3_blocks" --num-sbm 200 --num-nodes-range 300 500 --num-sbm-blocks 3
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_2_4_blocks/sbm_4_blocks" --num-sbm 200 --num-nodes-range 300 500 --num-sbm-blocks 4
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_6_8_blocks/sbm_6_blocks" --num-sbm 200 --num-nodes-range 300 500 --num-sbm-blocks 6
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_6_8_blocks/sbm_7_blocks" --num-sbm 200 --num-nodes-range 300 500 --num-sbm-blocks 7
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_6_8_blocks/sbm_8_blocks" --num-sbm 200 --num-nodes-range 300 500 --num-sbm-blocks 8



# GENERTE LAYOUTS
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_2_4_blocks/sbm_2_blocks" --layout-dir "layouts_2_4_blocks/sbm_2_blocks" 
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_2_4_blocks/sbm_3_blocks" --layout-dir "layouts_2_4_blocks/sbm_3_blocks" 
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_2_4_blocks/sbm_4_blocks" --layout-dir "layouts_2_4_blocks/sbm_4_blocks" 
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_6_8_blocks/sbm_6_blocks" --layout-dir "layouts_6_8_blocks/sbm_6_blocks" 
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_6_8_blocks/sbm_7_blocks" --layout-dir "layouts_6_8_blocks/sbm_7_blocks" 
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_6_8_blocks/sbm_8_blocks" --layout-dir "layouts_6_8_blocks/sbm_8_blocks" 



# EXECUTE TRAIN, TEST, PREDICT
python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_2_4_blocks" --layout-root-dir "layouts_2_4_blocks" \
                --predict-graph-root-dir "graphs_6_8_blocks" --predict-layout-root-dir "layouts_6_8_blocks" \
                --output-root-dir "outputs_6_8_blocks_FC" \
                --model-save-dir "model_train_2_4_blocks_FC" --log-dir "logs_train_2_4_blocks_FC" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs ${epochs} \
                --latent-dims 5 5 5 \
                --init-lr 0.0005

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_6_8_blocks" --layout-root-dir "layouts_6_8_blocks" \
                --predict-graph-root-dir "graphs_2_4_blocks" --predict-layout-root-dir "layouts_2_4_blocks" \
                --output-root-dir "outputs_2_4_blocks_FC" \
                --model-save-dir "model_train_6_8_blocks_FC" --log-dir "logs_train_6_8_blocks_FC" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs ${epochs} \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_2_4_blocks" --layout-root-dir "layouts_2_4_blocks" \
                --predict-graph-root-dir "graphs_6_8_blocks" --predict-layout-root-dir "layouts_6_8_blocks" \
                --output-root-dir "outputs_6_8_blocks_GCN" \
                --model-save-dir "model_train_2_4_blocks_GCN" --log-dir "logs_train_2_4_blocks_GCN" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs ${epochs} \
                --latent-dims 5 5 5 \
                --init-lr 0.0005

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_6_8_blocks" --layout-root-dir "layouts_6_8_blocks" \
                --predict-graph-root-dir "graphs_2_4_blocks" --predict-layout-root-dir "layouts_2_4_blocks" \
                --output-root-dir "outputs_2_4_blocks_GCN" \
                --model-save-dir "model_train_6_8_blocks_GCN" --log-dir "logs_train_6_8_blocks_GCN" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs ${epochs} \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_2_4_blocks" --layout-root-dir "layouts_2_4_blocks" \
                --predict-graph-root-dir "graphs_6_8_blocks" --predict-layout-root-dir "layouts_6_8_blocks" \
                --output-root-dir "outputs_6_8_blocks_latentgnn" \
                --model-save-dir "model_train_2_4_blocks_latentgnn" --log-dir "logs_train_2_4_blocks_latentgnn" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs ${epochs} \
                --latent-dims 5 5 5 \
                --init-lr 0.0005

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_6_8_blocks" --layout-root-dir "layouts_6_8_blocks" \
                --predict-graph-root-dir "graphs_2_4_blocks" --predict-layout-root-dir "layouts_2_4_blocks" \
                --output-root-dir "outputs_2_4_blocks_latentgnn" \
                --model-save-dir "model_train_6_8_blocks_latentgnn" --log-dir "logs_train_6_8_blocks_latentgnn" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs ${epochs} \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 



# ONLY PREDICT
# python main.py --experiment-root-dir ${ex_root_dir} \
#                 --graph-root-dir "graphs_2_4_blocks" --layout-root-dir "layouts_2_4_blocks" \
#                 --predict-graph-root-dir "graphs_6_8_blocks" --predict-layout-root-dir "layouts_6_8_blocks" \
#                 --output-root-dir "outputs_6_8_blocks" \
#                 --model-save-dir "model_train_2_4_blocks" --log-dir "logs_train_2_4_blocks" \
#                 --model-load-path "model_train_2_4_blocks/model.pt" \
#                 --position-encoding-dim ${ps_encoding_dim} --no-train --no-test 

# python main.py --experiment-root-dir ${ex_root_dir} \
#                 --graph-root-dir "graphs_6_8_blocks" --layout-root-dir "layouts_6_8_blocks" \
#                 --predict-graph-root-dir "graphs_2_4_blocks" --predict-layout-root-dir "layouts_2_4_blocks" \
#                 --output-root-dir "outputs_2_4_blocks" \
#                 --model-save-dir "model_train_6_8_blocks" --log-dir "logs_train_6_8_blocks" \
#                 --model-load-path "model_train_2_4_blocks/model.pt" \
#                 --position-encoding-dim ${ps_encoding_dim} --no-train --no-test 

