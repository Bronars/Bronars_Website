import numpy as np

from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer

def collect_random_trajectory(env, timesteps=1000):
    """Run a random policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
    """

    env.reset()
    dof = env.action_dim

    for t in range(timesteps):
        d = env._gripper1_to_handle1()
        action = np.random.randn(dof)
        env.step(action)
        env.render()
        if t % 100 == 0:
            print(t)
            print(d)

world = MujocoWorldBase()

mujoco_robot = Panda()

gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()

