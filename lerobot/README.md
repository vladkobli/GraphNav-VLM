## LeRobot
# Venv creation
```bash
chmod +x /home/vladkobli/rocon-demos/lerobot/rebuild_lerobot_venv.sh
/home/vladkobli/rocon-demos/lerobot/rebuild_lerobot_venv.sh
```

# Arm Calibration 
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lerobot
```
# Teleop
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lerobot \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=lerobot2
```

# Start server
```bash
source /home/vladkobli/rocon-demos/lerobot/.venv/bin/activate
python lerobot_camera_sweep_server.py
```