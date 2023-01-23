"""
Written by Danfei Xu

Recollect data by using a joint position controller to try and track the
reference trajectories.

This is useful for collecting datasets with lower control frequencies.
"""

import os
import shutil
import sys
import h5py
import json
import argparse
import random
import datetime
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

import mujoco_py
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import postprocess_model_xml

import RobotTeleop
from RobotTeleop.configs.base_config import BaseServerConfig

#import batchRL

# environment
# ENV_NAME = "SawyerCircusEasyTeleop"
ENV_NAME = "SawyerCircusTeleop"

# number of demos to collect
NUM_DEMOS = 27

# parameters for path
CORNER_DISTANCE = 0.2 # path is two sides of a triangle - this determines the corner
NUM_STEPS_LIFT = (40, 41) # task path length ranges
NUM_STEPS_REACH = (50, 51)
NUM_STEPS_INSERT = (50, 51)
NUM_STEPS_FIT = 50
NUM_STEPS_WAIT = 10
LIFT_X_RANGE = 0.01 # randomization for grasp
LIFT_Z_RANGE = 0.015
PERTURB = False

# CORNER_DISTANCE = 0.2 # path is two sides of a triangle - this determines the corner
# NUM_STEPS_LIFT = (40, 41) # task path length ranges
# NUM_STEPS_REACH = (50, 101)
# NUM_STEPS_INSERT = (50, 76)
# NUM_STEPS_FIT = 50
# NUM_STEPS_WAIT = 10
# LIFT_X_RANGE = 0.01 # randomization for grasp
# LIFT_Z_RANGE = 0.015
# PERTURB = True

def interpolate_poses(pos1, rot1, pos2, rot2, steps, perturb=False):
    """
    Extremely simple linear interpolation between two poses.

    If @perturb, randomly move all the interpolated position points in a uniform, non-overlapping grid.
    """

    delta_pos = pos2 - pos1
    pos_step_size = delta_pos / steps
    grid = np.arange(steps).astype(np.float64)
    if perturb:
        # move the interpolation grid points by up to a half-size forward or backward
        perturbations = np.random.uniform(
            low=-0.5,
            high=0.5,
            size=(steps - 2,),
        )
        grid[1:-1] += perturbations
    pos_steps = np.array([pos1 + grid[i] * pos_step_size for i in range(steps)])

    # break up axis angle delta rotation into smaller angles along same axis
    delta_rot_mat = rot1.dot(rot2.T)
    delta_quat = T.mat2quat(delta_rot_mat)
    delta_axis, delta_angle = T.quat2axisangle(delta_quat)
    rot_step_size = delta_angle / steps

    # convert into delta rotation matrices, and then convert to absolute rotations
    delta_rot_steps = [T.quat2mat(T.axisangle2quat(delta_axis, i * rot_step_size)) for i in range(steps)]
    rot_steps = np.array([delta_rot_steps[i].T.dot(rot1) for i in range(steps)])

    return pos_steps, rot_steps

def get_lift_waypoints(env, perturb=False):
    """
    Generate waypoints to grab the rod.
    """

    assert ENV_NAME == "SawyerCircusTeleop"

    # initial ee pose
    pos_0 = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id("right_hand")])
    rot_0 = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    
    # pose of handle
    pos_handle = np.array(env.sim.data.geom_xpos[env.sim.model.geom_name2id("block_handle")])
    rot_handle = np.array(rot_0) 

    # z-tolerance to hover above handle first
    pos_handle_hover = np.array(pos_handle)
    pos_handle_hover[2] += 0.04 #0.02

    # offset to correct for ee pos to center of ee gripper AND a little extra to center the grasp
    # since the site of the gripper is at the bottom, in between the finger edges
    ee_site = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("grip_site")])
    offset = pos_0 - ee_site
    offset[2] -= 0.02
    pos_handle += offset
    pos_handle_hover += offset

    if perturb:
        # sample a small offset to move the grasp location a little bit
        x_perturb = np.random.uniform(low=-LIFT_X_RANGE, high=LIFT_X_RANGE)
        z_perturb = np.random.uniform(low=-LIFT_Z_RANGE, high=LIFT_Z_RANGE)
        perturb = np.array([x_perturb, 0., z_perturb])
        pos_handle += perturb
        pos_handle_hover += perturb

    # sample random amount of time it should take to reach the handle
    num_steps_lift = np.random.randint(NUM_STEPS_LIFT[0], NUM_STEPS_LIFT[1])

    # hover above handle
    pos_steps0, rot_steps0 = interpolate_poses(
        pos1=pos_0, 
        rot1=rot_0, 
        pos2=pos_handle_hover, 
        rot2=rot_handle, 
        steps=num_steps_lift,
        perturb=PERTURB,
    )
    gripper_steps0 = [[-1.] for _ in range(num_steps_lift)] # open

    # reach handle
    pos_steps1, rot_steps1 = interpolate_poses(
        pos1=pos_handle_hover, 
        rot1=rot_handle, 
        pos2=pos_handle, 
        rot2=rot_handle, 
        steps=10,
        perturb=PERTURB,
    )
    gripper_steps1 = [[-1.] for _ in range(10)] # open

    all_pos_steps = np.concatenate([pos_steps0, pos_steps1])
    all_rot_steps = np.concatenate([rot_steps0, rot_steps1])
    all_grip_steps = np.concatenate([gripper_steps0, gripper_steps1])

    return all_pos_steps, all_rot_steps, all_grip_steps


def get_pose_waypoints(env):
    """
    Generate triangle path with interpolation.
    """

    # get the waypoints from current position of arm to goal
    pos_0 = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id("right_hand")])
    rot_0 = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    pos_goal, rot_goal, pos_goal2 = get_goal_pose(env)
    if ENV_NAME == "SawyerCircusEasyTeleop":
        # try for direct insertion instead of aligning tip in easy env
        pos_goal = pos_goal2

    # get vector in x-y plane parallel to the ring to find the x-y line that passes through the ring
    ring_geoms = []
    for i in range(env.num_ring_geoms):
        ring_geom_pos = np.array(env.sim.data.geom_xpos[env.sim.model.geom_name2id("hole_ring_{}".format(i))])
        ring_geoms.append(ring_geom_pos)
    ring_dir = ring_geoms[1][:2] - ring_geoms[0][:2]
    ring_perp = np.array([ring_dir[1], -ring_dir[0]])
    ring_perp /= np.linalg.norm(ring_perp)

    # place a corner waypoint a distance away from the hole
    pos_corner = np.array(pos_goal)
    pos_corner[0] += CORNER_DISTANCE * ring_perp[0]
    pos_corner[1] += CORNER_DISTANCE * ring_perp[1]
    rot_corner = np.array(rot_goal)

    # sample random amount of time it should take to traverse the triangle path
    num_steps = np.random.randint(NUM_STEPS_REACH[0], NUM_STEPS_REACH[1])

    # interpolated poses along the first leg of the triangle
    pos_steps, rot_steps = interpolate_poses(
        pos1=pos_0, 
        rot1=rot_0, 
        pos2=pos_corner, 
        rot2=rot_corner, 
        steps=num_steps,
        perturb=PERTURB,
    )
    gripper_steps = [[0.] for _ in range(num_steps)] # closed

    return pos_steps, rot_steps, gripper_steps


def get_insertion_waypoints(env):
    """
    Get waypoints for insertion. Assumes the arm is already sufficiently
    close to the hole.
    """

    # get the waypoints from current position of arm to goal
    pos_0 = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id("right_hand")])
    rot_0 = np.array(env.sim.data.body_xmat[env.sim.model.body_name2id("right_hand")].reshape([3, 3]))
    pos_goal, rot_goal, pos_goal2 = get_goal_pose(env)

    # sample random amount of time it should take to traverse the triangle path
    num_steps = np.random.randint(NUM_STEPS_INSERT[0], NUM_STEPS_INSERT[1])

    # interpolated poses
    pos_concat = []
    rot_concat = []
    gripper_concat = []

    pos_steps, rot_steps = interpolate_poses(
        pos1=pos_0, 
        rot1=rot_0, 
        pos2=pos_goal, 
        rot2=rot_goal, 
        steps=num_steps,
        perturb=PERTURB,
    )
    gripper_steps = [[0.] for _ in range(num_steps)] # closed
    pos_concat.append(pos_steps)
    rot_concat.append(rot_steps)
    gripper_concat.append(gripper_steps)

    if ENV_NAME == "SawyerCircusTeleop":
        pos_steps, rot_steps = interpolate_poses(
            pos1=pos_goal, 
            rot1=rot_goal, 
            pos2=pos_goal2, 
            rot2=rot_goal, 
            steps=NUM_STEPS_FIT,
            perturb=PERTURB,
        )
        gripper_steps = [[0.] for _ in range(NUM_STEPS_FIT)] # closed
        pos_concat.append(pos_steps)
        rot_concat.append(rot_steps)
        gripper_concat.append(gripper_steps)

    # add in fixed final poses as a buffer
    pos_steps = np.array([pos_goal2 for _ in range(NUM_STEPS_WAIT)])
    rot_steps = np.array([rot_goal for _ in range(NUM_STEPS_WAIT)])
    gripper_steps = [[0.] for _ in range(NUM_STEPS_WAIT)] # closed
    pos_concat.append(pos_steps)
    rot_concat.append(rot_steps)
    gripper_concat.append(gripper_steps)

    all_pos_steps = np.concatenate(pos_concat)
    all_rot_steps = np.concatenate(rot_concat)
    all_grip_steps = np.concatenate(gripper_concat)
    return all_pos_steps, all_rot_steps, all_grip_steps

def get_goal_pose(env):
    """
    Helper function to get goal location for arm based on the current state of the env.

    The function finds the relative transform between the tip of the rod and 
    the gripper, and uses that to find the correct gripper pose that will
    result in successful ring insertion.
    """

    # ring position is average of all the surrounding ring geom positions
    ring_pos = np.zeros(3)
    ring_geoms = []
    for i in range(env.num_ring_geoms):
        ring_geom_pos = np.array(env.sim.data.geom_xpos[env.sim.model.geom_name2id("hole_ring_{}".format(i))])
        ring_pos += ring_geom_pos
        ring_geoms.append(ring_geom_pos)
    ring_pos /= env.num_ring_geoms

    # get vector in x-y plane parallel to the ring to find the x-y line that passes through the ring
    ring_dir = ring_geoms[1][:2] - ring_geoms[0][:2]
    ring_perp = np.array([ring_dir[1], -ring_dir[0]])
    ring_perp /= np.linalg.norm(ring_perp)
    ring_angle = np.arctan2(ring_perp[1], ring_perp[0])

    # rotate the initial tip rotation by ring angle to get the target pose
    if ENV_NAME == "SawyerCircusTeleop":
        initial_tip_mat = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    else:
        initial_tip_mat = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    target_tip_mat = T.rotation_matrix(angle=-ring_angle, direction=[0., 0., 1.])[:3, :3].T.dot(initial_tip_mat)
    target_tip_pose = T.make_pose(ring_pos, target_tip_mat)

    # gripper pose
    gripper_pose = T.make_pose(
        np.array(env.sim.data.body_xpos[env.sim.model.body_name2id("right_hand")]),
        np.array(env.sim.data.body_xmat[env.sim.model.body_name2id("right_hand")].reshape([3, 3])),
    )

    # tip pose
    if ENV_NAME == "SawyerCircusTeleop":
        block_pos = np.array(env.sim.data.geom_xpos[env.sim.model.geom_name2id("block_thread")])
        block_pose = np.array(env.sim.data.geom_xmat[env.sim.model.geom_name2id("block_handle")].reshape(3, 3))
        handle_pos = np.array(env.sim.data.geom_xpos[env.sim.model.geom_name2id("block_handle")])
    else:
        # the handle is basically the robot gripper
        block_pos = np.array(env.sim.data.body_xpos[env.object_body_ids["r_gripper_rod"]])
        block_pose = np.array(env.sim.data.body_xmat[env.object_body_ids["r_gripper_rod"]].reshape(3, 3))
        handle_pos = np.array(gripper_pose[3, :3])
        handle_pos[2] = block_pos[2]
    block_dir = block_pos - handle_pos
    block_dir /= np.linalg.norm(block_dir)
    tip_pos = block_pos + 0.06 * block_dir
    tip_pose = T.make_pose(tip_pos, block_pose)
    world_pose_in_tip = T.pose_inv(tip_pose)
    block_pose = T.make_pose(block_pos, block_pose)
    world_pose_in_block = T.pose_inv(block_pose)

    # arm pose in tip frame and in block frame (center of rod)
    arm_in_tip = T.pose_in_A_to_pose_in_B(gripper_pose, world_pose_in_tip)
    arm_in_block = T.pose_in_A_to_pose_in_B(gripper_pose, world_pose_in_block)

    # use the relative transform between tip / block and arm to get target gripper poses
    target_gripper_pose1 = T.pose_in_A_to_pose_in_B(arm_in_tip, target_tip_pose)
    target_gripper_pose2 = T.pose_in_A_to_pose_in_B(arm_in_block, target_tip_pose)

    return target_gripper_pose1[:3, 3], target_gripper_pose1[:3, :3], target_gripper_pose2[:3, 3]


def get_robosuite_env(render):
    """
    Constructs robosuite env
    """

    # we use specific OSC controller args in RobotTeleop
    controller_json_path = os.path.join(RobotTeleop.__path__[0], 
            "assets/osc/robosuite/osc.json")
    with open(controller_json_path, "r") as f:
        controller_args = json.load(f)

    osc_args = dict(
        use_camera_obs=False,
        reward_shaping=False,
        gripper_visualization=False,
        has_renderer=render,
        has_offscreen_renderer=False,
        control_freq=20,
        ignore_done=True,
        eval_mode=True,
        perturb_evals=False,
        controller_config=controller_args,
        # use_indicator_object=True,
        # indicator_num=2,
    )
    env = robosuite.make(ENV_NAME, **osc_args)

    return env


def get_fake_teleop_config():
    """
    Helper function to fake a teleoperation config. This is used for parsing
    env arguments, among other things, during postprocessing and training.
    """
    config = BaseServerConfig()
    config.robot.type = "MujocoSawyerRobot"
    config.robot.task.name = ENV_NAME
    config.controller.mode = "osc"
    config.data_collection.dagger_mode = False
    config.infer_settings()
    return config

def follow_waypoints(env, ref_ee_pos, ref_ee_mat, ref_ee_grip, render=False):
    """
    Helper function to follow a sequence of waypoints with OSC control, and
    return the seqeunce of states and actions visited.
    """

    # store controller bounds for re-scaling actions
    MAX_DPOS = env.controller.output_max[0]
    MAX_DROT = env.controller.output_max[3]

    states = []
    actions = []
    success = False
    for j in range(len(ref_ee_pos)):

        # current mjstate
        state = np.array(env.sim.get_state().flatten())

        desired_position = ref_ee_pos[j]
        current_position = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id("right_hand")])
        delta_position = desired_position - current_position
        delta_position = np.clip(delta_position / MAX_DPOS, -1., 1.)

        # use the OSC controller's convention for delta rotation
        desired_rotation = ref_ee_mat[j]
        current_rotation = np.array(env.sim.data.body_xmat[env.sim.model.body_name2id("right_hand")].reshape([3, 3]))
        delta_rot_mat = current_rotation.dot(desired_rotation.T)
        delta_quat = T.mat2quat(delta_rot_mat)
        delta_axis, delta_angle = T.quat2axisangle(delta_quat)
        delta_rotation = -T.axisangle2vec(delta_axis, delta_angle)
        delta_rotation = np.clip(delta_rotation / MAX_DROT, -1., 1.)

        # play the action
        play_action = np.concatenate([delta_position, delta_rotation, ref_ee_grip[j]])
        env.step(play_action)
        if render:
            env.render()

        # collect data
        states.append(state)
        actions.append(play_action)

        # termination condition
        done = int(env._check_success())
        success = success or done
        # if success:
        #     break

    return states, actions, success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=str,
    )
    parser.add_argument(
        "--render",
        action='store_true', # to visualize collection
    )
    args = parser.parse_args()

    env = get_robosuite_env(render=args.render)
    fake_teleop_config = get_fake_teleop_config()

    # file to write
    f = h5py.File(args.batch, "w")
    grp = f.create_group("data")

    avg_success = 0.
    num_failures = 0
    ep_lengths = []
    demo_count = 0
    num_demos_attempted = 0
    for i in range(NUM_DEMOS):

        # store trajectory under new group
        ep_data_grp = grp.create_group("demo_{}".format(demo_count))

        # start new ep, using the hack for deterministic playback
        env.reset()
        initial_mjstate = env.sim.get_state().flatten()
        xml = env.model.get_xml()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_mjstate)
        env.sim.forward()

        if args.render:
            env.viewer.set_camera(2)

        if ENV_NAME == "SawyerCircusTeleop":
            # first, grab the rod, so that we can obtain the relative pose of the rod to the arm
            # and use that to set the goal
            ref_ee_pos, ref_ee_mat, ref_ee_grip = get_lift_waypoints(
                env=env,
                perturb=PERTURB,
            )
            lift_states, lift_actions, _ = follow_waypoints(
                env=env, 
                ref_ee_pos=ref_ee_pos, 
                ref_ee_mat=ref_ee_mat, 
                ref_ee_grip=ref_ee_grip, 
                render=args.render,
            )


        # first leg of triangle        
        ref_ee_pos, ref_ee_mat, ref_ee_grip = get_pose_waypoints(env)
        states, actions, _ = follow_waypoints(
            env=env, 
            ref_ee_pos=ref_ee_pos, 
            ref_ee_mat=ref_ee_mat, 
            ref_ee_grip=ref_ee_grip, 
            render=args.render,
        )

        # get insertion trajectory based on where we end up after the first part of task
        ref_ee_pos, ref_ee_mat, ref_ee_grip = get_insertion_waypoints(env)
        insertion_states, insertion_actions, success = follow_waypoints(
            env=env, 
            ref_ee_pos=ref_ee_pos, 
            ref_ee_mat=ref_ee_mat, 
            ref_ee_grip=ref_ee_grip, 
            render=args.render,
        )

        avg_success += float(success)
        num_demos_attempted += 1

        # skip failures
        if not success:
            del grp["demo_{}".format(demo_count)]
            continue

        if ENV_NAME == "SawyerCircusTeleop":
            # prepend the lifting states and actions
            states = lift_states + states
            actions = lift_actions + actions
        states += insertion_states
        actions += insertion_actions

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

        # write model file
        ep_data_grp.attrs["model_file"] = xml

        demo_count += 1
        print("{} successes out of {}".format(avg_success, num_demos_attempted))

    print("Done!\n")
    print("{} successes out of {}".format(avg_success, num_demos_attempted))
    print("{} exceptions out of {}\n".format(num_failures, num_demos_attempted))

    avg_success /= num_demos_attempted

    print("Average Success: {}".format(avg_success))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["env"] = ENV_NAME
    grp.attrs["teleop_config"] = fake_teleop_config.dump()

    f.close()