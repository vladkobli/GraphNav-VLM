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

## Add descriptions to existing json
After having dataset created and attributed the neighbours of each node
```bash
python add_descriptions.py \
  --graph-json nodes/nodes_graph_8_color.json \
  --dataset-root nodes \
  --output nodes_graph_8_color_described.json
```

## Run Experiment on offline map
```bash
## DFS vs Visual
# DFS
python run_nav_experiment.py \
  --graph-json nodes_graph_8_color_described.json \
  --model qwen2.5:7b \
  --task "Go indoors" \
  --start-node n44 \
  --goal-nodes n19 \
  --max-steps 60 \
  --memory-mode partial \
  --controller-mode dfs \
  --manual-path n44,n43,n42,n41,n24,n20,n19 \
  --output-dir results

# Visual
python run_nav_experiment.py \
  --graph-json nodes_graph_8_color_described3.json \
  --model qwen2.5:7b \
  --task "Find a bike rack" \
  --start-node n12 \
  --goal-nodes n44,n45,n46 \
  --max-steps 60 \
  --memory-mode partial \
  --controller-mode visual \
  --model-stop-on-target \
  --stop-confidence-threshold 0.75 \
  --visual-max-node-visits 2 \
  --visual-max-edge-repeats 1 \
  --manual-path n12,n11,n16,n19,n20,n24,n41,n42,n43,n45 \
  --output-dir results
```