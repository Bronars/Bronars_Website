"""
Record trajectory data with the DataCollectionWrapper wrapper and play them back.

Example:
    $ python demo_collect_and_playback_data.py --environment Lift
"""

import argparse
import os
from glob import glob

import numpy as np

import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper


def get_controller_action(env):
    """Run a random policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
    """

    # Define neutral value
    action = np.zeros(2*action_dim + 2*env.robots[0].gripper.dof)
    arm0_dist = env._gripper0_to_cubeA
    arm1_dist = env._gripper1_to_cubeB
    action[0:3] = arm0_dist[:]
    action[2] = 0
    action[7:10] = arm1_dist[:]
    action[9] = 0
    
    # Set real actions
    #action[4:7] = env._handle1_xpos
    #action[0:3] = arm0_dist
    
    #print(np.linalg.norm(arm0_dist))
    
    if arm0_dist[0] < .015 and arm0_dist[1] < 0.015:
        action[0:3] = arm0_dist + env.cubeA.bottom_offset
    
    
    if abs(arm0_dist[2]) < .03:
        print(arm0_dist[2])
        action[6] = 1
    else:
        action[6] = 0
    
    if env._check_grasp(env.robots[0].gripper, env.cubeA):
            action[0:3] = [0, 0, .1]
    
    if arm1_dist[0] < .015 and arm1_dist[1] < 0.015:
        action[7:10] = arm1_dist + env.cubeB.bottom_offset
    
    
    if abs(arm1_dist[2]) < .03:
        action[13] = 1
    else:
        action[13] = 0
    
    if env._check_grasp(env.robots[0].gripper, env.cubeB):
            action[7:10] = [0, 0, .1]
    
    return action
    if np.linalg.norm(arm1_dist) < .04:
        action[13] = 1
    action[7:10] = arm1_dist
    action[2] = action[2] + .5
    action[9] = action[9] + .1

    print(action)
    return action


# Hacky way to grab joint dimension for now
joint_dim = 6

# Choose controller
controller_name = "OSC_POSE"

# Load the desired controller
controller_configs = suite.load_controller_config(default_controller=controller_name)

# Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value)
controller_settings = {
    "OSC_POSE": [6, 6, 0.1],
    "OSC_POSITION": [3, 3, 0.1],
    "IK_POSE": [6, 6, 0.01],
    "JOINT_POSITION": [joint_dim, joint_dim, 0.2],
    "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
    "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
}

# Define variables for each controller test
action_dim = controller_settings[controller_name][0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--environment", type=str, default="TwoArmLift")
    #parser.add_argument("--robots", nargs="+", type=str, default=["Panda", "Panda"], help="Which robot(s) to use in the env")
    parser.add_argument("--directory", type=str, default="/tmp/")
    parser.add_argument("--timesteps", type=int, default=500)
    args = parser.parse_args()

    # create original environment
    env = suite.make(
    "TwoArmLift",
    robots=["Panda", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_configs,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=500,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
)
    data_directory = args.directory



    # wrap the environment with data collection wrapper
    #env = DataCollectionWrapper(env, data_directory)

    # testing to make sure multiple env.reset calls don't create multiple directories
    #env.reset()
    #env.reset()

    # collect some data
    
    
    count = 0
    holding = False
    while count < args.timesteps:
        action = get_controller_action(env)
        env.step(action)
        env.render()
        count +=1


    env.close
        