# GraphNav 

## 1. Setup
```bash
# clone repo
cd /rgbd_camera_intel_dev/src
git clone ...

# Build packages
cd /rgbd_camera_intel_dev
colcon build
source install/setup.bash
```

## 2. Launch 
1. Launching LeRobot Server
```bash
cd /home/vladkobli/rocon-demos/lerobot
source /home/vladkobli/rocon-demos/lerobot/.venv/bin/activate
python lerobot_camera_sweep_server.py
```

2. Launching Moondream Server
```bash
cd /home/vladkobli/rocon-demos/moondream
source /home/vladkobli/rocon-demos/moondream/venv_moondream/bin/activate

python /home/vladkobli/rocon-demos/jazzy-full/src/graph_nav/graph_nav/add_description2.py \
  --server \
  --host 0.0.0.0 \
  --port 8766 \
  --path-map /rgbd_camera_intel_dev/src=/home/vladkobli/rocon-demos/jazzy-full/src
```

3. Starting Qwen server
```bash
cd /home/vladkobli/rocon-demos/moondream
ollama serve
ollama pull qwen2.5:7b
```

4. Launching the pipeline, specifying the user task as an argument
```bash
ros2 launch graph_nav graph_nav_full.launch.py \
  lerobot_server_url:=http://127.0.0.1:8765 \
  moondream_server_url:=http://127.0.0.1:8766 \
  user_prompt:="Find a car" \
  target_prompt:=car \
  moondream_overwrite_descriptions:=true \
  publish_nav2_goal_pose:=true \
  next_node_goal_distance_m:=2.0

# Including LLM
ros2 launch graph_nav graph_nav_full.launch.py \
  lerobot_server_url:=http://127.0.0.1:8765 \
  moondream_server_url:=http://127.0.0.1:8766 \
  ollama_base_url:=http://127.0.0.1:11434 \
  llm_model:=qwen2.5:7b \
  user_prompt:="Find a car" \
  target_prompt:=car \
  moondream_overwrite_descriptions:=true \
  publish_nav2_goal_pose:=true \
  next_node_goal_distance_m:=2.0 \
  llm_candidate_selection_enabled:=true \
  llm_request_timeout_sec:=60.0 \
  llm_candidate_num_predict:=700
```