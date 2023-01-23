"""
A script for adding randomization to dataset trajectories by loading the 
first state and playing actions back (with added noise) open-loop.
The script can generate videos as well, by rendering simulation frames
during playback.

Currently the script is hard coded to work for two arm tasks (panda-panda)
with OSC Pose controllers.  This only affects randomize_actions function

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key
        
    result_path (str): if provided, save results to this file path
    
    rand_mag (float): adds random noise in range(-rand_mag, rand_mag) to the actions 
        sent to the controller (currently only tested on position controllers).  Should
        be on the order of .01 if generating sucessful trajectories
    
    stable_frq (int): number of actions from original demonstration that you
        want to remain unchanged.  Higher number makes sucessful trajectories
        more frequent
    
    stable_type (int): 0 means that stable time points are evenly distributed through the
        timesteps.  1 means that the stable time points are randomly distributed

    stable_buffer (int): number of frames that should remain unchanged at each position 
        that is selected as stable.  Actually allows the controller to converge to the 
        desired location if the buffer is large enough

    stable_grip (int): number of frames that should remain unchanged when the gripper is
        activated/deactivated.  Makes the gripper more likely to actually grap the object


Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import argparse
import imageio
import time
import numpy as np
from copy import deepcopy

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

def randomize_actions(actions, mag, stable_frq, stable_type, stable_buff, stable_grip):
    gripper_change1 = np.where(np.diff(np.sign(actions[:,6])))[0]
    gripper_change2 = np.where(np.diff(np.sign(actions[:,-1])))[0]
    np.random.seed(int(time.time()))
    s1, s2 = actions.shape
    randomizer = np.random.uniform(low = -mag, high = mag, size = (s1, s2))
    #randomizer[:, 6] = 0
    #randomizer[:, -1] = 0
    if stable_type == 0:
        frq = np.arange(0, actions.shape[0], stable_frq)
    else:
        frq = np.random.randint(actions.shape[0], size = stable_frq)
    frq[frq > actions.shape[0] - 11] = actions.shape[0] - 11
    gripper_change1[gripper_change1 > actions.shape[0] - 11] = actions.shape[0] - 11
    gripper_change2[gripper_change2 > actions.shape[0] - 11] = actions.shape[0] - 11
    for x in range(int(stable_buff/2)):
        randomizer[frq - x] = 0
        randomizer[frq + x] = 0
    for x in range(int(stable_grip/2)):
        randomizer[gripper_change1 - x] = 0
        randomizer[gripper_change2 + x] = 0
    #randomizer[-stable_grip:-1, :] = 0
    result = randomizer + actions
    result[result > 1 ] = 1
    result[result < -1] = -1
    return result


def playback_trajectory_with_env(
    env, 
    initial_state, 
    old_states, 
    actions=None, 
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        old_states=np.array(old_states),
        states = [], 
        initial_state_dict=initial_state,
    )

    traj_len = old_states.shape[0]
    
    assert old_states.shape[0] == actions.shape[0]

    #divergence = []

    for i in range(traj_len):

        next_obs, _, _, _ = env.step(actions[i])
        if i < traj_len - 1:
            # Get original state for divergance comparisons
            state_playback = env.get_state()["states"]
            if not np.all(np.equal(old_states[i + 1], state_playback)):
                err = np.linalg.norm(old_states[i + 1] - state_playback)
                #divergence.append(err)
                print("playback diverged by {} at step {}".format(err, i))

        r = env.get_reward()
        done = int(env.is_success()["task"])
        
        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["states"].append(state_playback)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    print("Successful?: " + str(sum(traj["rewards"])))

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def playback_dataset(args):
    
    #Add randomization parameters to the name so that they are not forgotten
    name_addition = "_" + "m" + str(args.rand_mag) + "f" + str(args.stable_frq) + "t" + str(args.stable_type) + "b" + str(args.stable_buffer) + "g" + str(args.stable_grip)
    name_addition = name_addition.replace('.', '')

    #-5 assumes that the file extension is .hdf5 (which it should be)
    file_path = args.result_path[:-5] + name_addition + args.result_path[-5:]
   
    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta)#, render=args.render, render_offscreen=write_video)

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    f_out = h5py.File(file_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(file_path))

        
    total_samples = 0
    divergences = {}
    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        #Currently hard coded for OSC-POSE contoller

        #Randomizing actions
        actions = f["data/{}/actions".format(ep)][()]
        actions = randomize_actions(actions, args.rand_mag, args.stable_type, args.stable_frq, args.stable_buffer, args.stable_grip)
        

        traj = playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            old_states=states, actions=actions, 
        )
        #div_key = "demo" + str(ind)
        #divergences[div_key] = divergence

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), file_path))
    f.close()
    f_out.close()
    
    # with open(file_path + ".json", "w") as outfile:
    #     json.dump(divergences, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Dataset that you want to be randomized
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="save results to this file path, must end in .hdf5",
    )

    #The rest of the acguments are randomization paramete
    parser.add_argument(
        "--rand_mag",
        type = float,
        default=.02,
        help="adds random noise with magnitude n",
    )
    parser.add_argument(
        "--stable_frq",
        type = int,
        default = 25,
        help="determines how many actions in trajectory should left un-randomized",
    )
    parser.add_argument(
        "--stable_buffer",
        type = int,
        default = 6,
        help="how many frames of un-randomized actions should be unrandomized",
    )
    parser.add_argument(
        "--stable_type",
        type = int,
        default = 1,
        help="0 - uniform distribution, 1 - random",
    )
    parser.add_argument(
        "--stable_grip",
        type = int,
        default = 10,
        help="number of frames around gripper activation/deactivation that should be unrandomized",
    )

    args = parser.parse_args()
    playback_dataset(args)