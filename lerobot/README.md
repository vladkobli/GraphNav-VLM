## LeRobot
# Venv creation
```bash
chmod +x setup/rebuild_lerobot_venv.sh
./setup/rebuild_lerobot_venv.sh
```

# Arm Calibration 
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lerobot
```

# Start server
```bash
source .venv/bin/activate
python lerobot_camera_sweep_server.py
```