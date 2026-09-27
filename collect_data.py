import numpy as np
import gymnasium as gym
import f1tenth_gym

def gap_follower(scan):
    """need to only focus on whats in front 
    - ignore infinite lidar readings
    - find where we can see the largest gap that isnt zero
    - compute steering angle and speed
    """

    "takes in scan of outside world that we have to clean"

    #turn raw scan into values from 0-10, with 0 being the closest and 10 being the furthest away
    clean_scan = np.nan_to_num(scan, nan = 0, posinf = 10, neginf = 0)
    clean_scan = np.clip(clean_scan, 0, 10)

    #total number of points we have scans to
    num_points = len(clean_scan)

    #should provide the middle 40% of the 270 degree scan 
    start_angle = int(num_points * 0.3)
    end_angle = int(num_points * 0.7)
    front = clean_scan[start_angle: end_angle]

    #choose which direction to go bsed  on angles in front of the car
    best_angle = np.argmax(front) + start_angle

    #calculate how far off the best_angle is from the center:
    #this isn ormalized between -1 and 1
    scaled_offset = ((best_angle - 324)/(756-324)) *2 - 1

    #calc optimized steerin angle based on the offset:
    steering_angle = np.clip(scaled_offset * .35, -.41, .41)

    #speed cotnrol
    min_dist_ahead = np.min(front)
    if(min_dist_ahead < 1.5 or abs(steering_angle) > 0.2):
       speed = 2
    else:
        speed = 4

    return steering_angle, speed

def main():
    # Initialize single car environment on Spielberg
    env = gym.make(
        "f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
        }
    )

    obs, info = env.reset()

    # Buffers to store our training pairs
    scans_data = []
    actions_data = []

    total_steps = 3000
    print(f"Starting data collection for {total_steps} steps...")

    for step in range(total_steps):
        # Current 1080-point LiDAR scan for the car
        current_scan = obs["scans"][0]

        # Get control commands from teacher
        steer, speed = gap_follower(current_scan)

        # Store input (scan) and label/target (steer, speed)
        scans_data.append(current_scan)
        actions_data.append([steer, speed])

        # Step the car forward
        action = np.array([[steer, speed]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Handle crashes or episode ends
        if obs["collisions"][0] == 1.0 or terminated or truncated:
            print(f"Collision or reset at step {step}! Resetting environment...")
            obs, info = env.reset()

        if (step + 1) % 500 == 0:
            print(f"Step {step + 1}/{total_steps} completed")

    # Convert to NumPy arrays and save as .npz
    scans_arr = np.array(scans_data, dtype=np.float32)
    actions_arr = np.array(actions_data, dtype=np.float32)

    np.savez("f1tenth_dataset.npz", scans=scans_arr, actions=actions_arr)
    print("\nData collection finished!")
    print(f"Saved 'f1tenth_dataset.npz' with:")
    print(f" - scans shape:   {scans_arr.shape}")
    print(f" - actions shape: {actions_arr.shape}")

if __name__ == "__main__":
    main()
    


