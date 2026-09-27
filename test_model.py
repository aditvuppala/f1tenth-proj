import gymnasium as gym
import f1tenth_gym
import numpy as np
import torch
import torch.nn as nn

class DrivingPolicyNet(nn.Module):
    def __init__(self, input_dim=1080, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading policy on device: {device}...")

    # Load model and weights
    model = DrivingPolicyNet().to(device)
    model.load_state_dict(torch.load("f1tenth_policy.pth", map_location=device))
    model.eval()  # Disables dropout

    # Initialize environment
    env = gym.make(
        "f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
        }
    )

    obs, info = env.reset()

    total_steps = 3000
    collision_count = 0
    print(f"Running autonomous policy for {total_steps} steps...")

    with torch.no_grad():
        for step in range(total_steps):
            # Extract raw scan
            raw_scan = obs["scans"][0]

            # Preprocessing: clean and normalize [0, 10] -> [0, 1]
            clean_scan = np.nan_to_num(raw_scan, nan=0.0, posinf=10.0, neginf=0.0)
            clean_scan = np.clip(clean_scan, 0.0, 10.0)
            norm_scan = clean_scan / 10.0

            # Convert to PyTorch tensor (batch shape: [1, 1080])
            scan_tensor = torch.tensor(norm_scan, dtype=torch.float32).unsqueeze(0).to(device)

            # Neural network inference
            action_pred = model(scan_tensor).squeeze(0).cpu().numpy()

            # Action bounds safety clipping
            steer = np.clip(action_pred[0], -0.41, 0.41)
            speed = np.clip(action_pred[1], 0.5, 5.0)

            # Step environment
            action = np.array([[steer, speed]], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            # Collision tracking
            if obs["collisions"][0] == 1.0 or terminated or truncated:
                collision_count += 1
                print(f"Collision at step {step}! Resetting...")
                obs, info = env.reset()

            if (step + 1) % 500 == 0:
                print(f"Step {step + 1}/{total_steps} completed")

    print("\n--- Autonomous Test Complete ---")
    print(f"Total steps simulated: {total_steps}")
    print(f"Total collisions:      {collision_count}")
    print(f"Collision rate:        {(collision_count / total_steps) * 100:.2f}%")

if __name__ == "__main__":
    evaluate()