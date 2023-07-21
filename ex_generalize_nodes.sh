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

ex_root_dir="ex_layout_2_dims/ex_generalize_nodes/ex1/"
ps_encoding_dim=30

# GENERATE GRAPHS
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_n_80_200/sbm" --num-sbm 300 --num-nodes-range 80 200
python graph_generate.py --experiment-root-dir ${ex_root_dir} --save-dir-sbm "graphs_n_500_600/sbm" --num-sbm 300 --num-nodes-range 500 600



# # GENERTE LAYOUT
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_n_80_200/sbm" --layout-dir "layouts_n_80_200/sbm" 
python neulay_2.py --experiment-root-dir ${ex_root_dir} --graph-dir "graphs_n_500_600/sbm" --layout-dir "layouts_n_500_600/sbm" 



# EXECUTE TRAIN, TEST, PREDICT
python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_n_80_200" --layout-root-dir "layouts_n_80_200" \
                --predict-graph-root-dir "graphs_n_500_600" --predict-layout-root-dir "layouts_n_500_600" \
                --output-root-dir "outputs_n_500_600_FC" \
                --model-save-dir "model_train_80_200_FC" --log-dir "logs_train_80_200_FC" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs 100 \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_n_500_600" --layout-root-dir "layouts_n_500_600" \
                --predict-graph-root-dir "graphs_n_80_200" --predict-layout-root-dir "layouts_n_80_200" \
                --output-root-dir "outputs_n_80_200_FC" \
                --model-save-dir "model_train_500_600_FC" --log-dir "logs_train_500_600_FC" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs 100 \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 



python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_n_80_200" --layout-root-dir "layouts_n_80_200" \
                --predict-graph-root-dir "graphs_n_500_600" --predict-layout-root-dir "layouts_n_500_600" \
                --output-root-dir "outputs_n_500_600_GCN" \
                --model-save-dir "model_train_80_200_GCN" --log-dir "logs_train_80_200_GCN" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs 100 \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_n_500_600" --layout-root-dir "layouts_n_500_600" \
                --predict-graph-root-dir "graphs_n_80_200" --predict-layout-root-dir "layouts_n_80_200" \
                --output-root-dir "outputs_n_80_200_GCN" \
                --model-save-dir "model_train_500_600_GCN" --log-dir "logs_train_500_600_GCN" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs 100 \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 



python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_n_80_200" --layout-root-dir "layouts_n_80_200" \
                --predict-graph-root-dir "graphs_n_500_600" --predict-layout-root-dir "layouts_n_500_600" \
                --output-root-dir "outputs_n_500_600_latentgnn" \
                --model-save-dir "model_train_80_200_latentgnn" --log-dir "logs_train_80_200_latentgnn" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs 100 \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 

python main.py --experiment-root-dir ${ex_root_dir} \
                --graph-root-dir "graphs_n_500_600" --layout-root-dir "layouts_n_500_600" \
                --predict-graph-root-dir "graphs_n_80_200" --predict-layout-root-dir "layouts_n_80_200" \
                --output-root-dir "outputs_n_80_200_latentgnn" \
                --model-save-dir "model_train_500_600_latentgnn" --log-dir "logs_train_500_600_latentgnn" \
                --model "latentgnn" \
                --position-encoding-dim ${ps_encoding_dim} --epochs 100 \
                --latent-dims 5 5 5 \
                --init-lr 0.0005 