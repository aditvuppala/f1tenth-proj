import numpy as np

def gap_follower(scan):
    """need to only focus on whats in front 
    - ignore infinite lidar readings
    - find where we can see the largest gap that isnt zero
    - compute steering angle and speed
    """

    "takes in scan of outside world that we have to clean"

    #turn raw scan into values from 0-10, with 0 being the closest and 10 being the furthest away
    clean_scan = np.nan_to_num(scan, NaN = 0, posinf = 10, neginf = 0)
    clean_scan = np.clip(clean_scan, 0, 10)

    #total number of points we have scans to
    num_points = len(clean_scan)

    #should provide the middle 40% of the 270 degree scan 
    start_angle = int(num_points * 0.3)
    end_angle = int(num_points * 0.7)
    front = clean_scan[start_angle: end_angle]

    #choose which direction to go bsed  on angles in front of the car
    best_angle = np.max(front) + start_angle

    #calculate how far off the best_angle is from the center:
    #this isn ormalized between -1 and 1
    scaled_offset = ((best_angle - 324)/(756-321)) *2 - 1

    #calc optimized steerin angle based on the offset:
    steering_angle = np.clip(scaled_offset * .35, -.41, .41)

    #speed cotnrol
    min_dist_ahead = np.min(front)
    if(min_dist_ahead < 1.5 or abs(steering_angle > 0.2)):
       speed = 2
    else:
        speed = 4

    return steering_angle, speed
    


