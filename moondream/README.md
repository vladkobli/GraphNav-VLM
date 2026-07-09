# Moondream
Describing image sequences from nodes using Moondream

## Build venv
```bash
# Build venv
cd ~/rocon-demos/moondream/
./setup/rebuild_moondream_env.sh

# Source venv
source ~/rocon-demos/moondream/venv_moondream/bin/activate
```

## Create Dataset
Follow initial steps in `camera_capture` package

After having dataset created and attributed the neighbours of each node
```bash
python add_descriptions.py \
  --graph-json nodes/nodes_graph_8_color.json \
  --dataset-root nodes \
  --output nodes_graph_8_color_described.json
```

## Run Experiment on offline map
```bash
# DFS
python run_nav_experiment.py \
    --graph-json nodes_graph_8_color_described.json \
    --model qwen2.5:7b \
    --task "Find a bike rack" \
    --start-node "n37" \
    --goal-nodes "n57,n58,n59,n60,n61,n62,n64,n65" \
    --max-steps "30" \
    --memory-mode partial \
    --controller-mode dfs \
    --manual-path "n37,n36,n35,n32,n33,n34,n57" \
    --output-dir "results_dfs"

# Visual
python run_nav_experiment.py \
    --graph-json nodes_graph_8_color_described.json \
    --model qwen2.5:7b \
    --task "Find a bike rack" \
    --start-node "n37" \
    --goal-nodes "n57,n58,n59,n60,n61,n62,n64,n65" \
    --max-steps "30" \
    --memory-mode partial \
    --controller-mode visual \
    --model-stop-on-target \
    --stop-confidence-threshold 0.75 \
    --visual-max-node-visits 2 \
    --visual-max-edge-repeats 1 \
    --manual-path "n37,n36,n35,n32,n33,n34,n57" \
    --output-dir "results_visual"
```