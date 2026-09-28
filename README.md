**F1TENTH Autonomous Racing: Imitation Learning Pipeline** \
A small behavior-cloning project in the F1TENTH Gym simulator. A PyTorch network learns to map a LiDAR scan to a steering angle and speed by imitating a hand-written driving controller, then drives the car closed-loop in simulation. 

Pipline:

1. Collect data: A gap-following agent drives the circuit, logging LiDAR scans along with steering angle and speed data. 
2. Train: A connected network is trained to rpedic the controllers steering and speed from the scan. 
3. Evaluate: The trained network drives the car in simulation and collisions are counted. 


Uses Python 3.12 and PyTorch
