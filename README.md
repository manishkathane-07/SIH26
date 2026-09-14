# SIH26 PS 26126 — VISTA-UGV Advanced Prototype

Vision-first autonomous navigation prototype for a GPS-denied outdoor UGV.

Pipeline:
Camera -> YOLO segmentation/detection -> free-space estimation -> monocular depth
proxy -> visual odometry -> local occupancy/cost map -> dynamic-window trajectory scoring
-> collision-risk supervisor -> steering/throttle command.

The prototype is software-only and can run from a webcam, RTSP/MJPEG stream, or MP4.
It is intended for SIH demonstration, not direct deployment on a real vehicle.

## Windows
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py

On first run YOLO weights are downloaded automatically.

Recommended demo:
1. Use a prerecorded outdoor road/field video as the primary demo.
2. Set a virtual destination on the right/left side.
3. Show obstacle boxes/masks, free-space mask, local costmap, trajectory candidates,
   visual odometry, risk score and command.
4. Enable "Autonomous mode" only in simulation/video demo.

## Important limitation
A monocular camera does not provide metric scale by itself. Visual odometry is therefore
relative unless calibrated with known camera height/IMU/wheel odometry. The dashboard
labels this explicitly.

## Suggested final hardware upgrade
Raspberry Pi 5 / Jetson Orin Nano + CSI camera + IMU + wheel encoders + motor controller.
Use the same perception/planning interfaces and replace the demo command sink with a
ROS 2 / serial / CAN motor driver.
