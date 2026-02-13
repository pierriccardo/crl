"""
Test VariBAD on Ant Direction env. Uses the AntDir env below; training part imports VariBAD
and runs the same training loop as varibad.py (no continual wrapper).
"""
import os
import random
import wandb
import numpy as np
import torch
import tyro
from tqdm import tqdm

from dataclasses import asdict
import matplotlib.pyplot as plt

from crl.algos.varibad import Config as VaribadConfig, VariBAD
from crl.algos.ppo import Config as PPOConfig
from crl.envs import EnvConfig, get_env_dims, make_env
from crl.buffers import SimpleTrajBuffer

"""
Based on rllab's serializable.py file

https://github.com/rll/rllab
"""

import inspect
import sys

ant_xml ="""<!-- Same a gym ant but with sites -->
<mujoco model="ant">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option integrator="RK4" timestep="0.01"/>
    <custom>
        <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
    </custom>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
    </default>
    <asset>
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01"
                 rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3"
               specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1"
              size="40 40 40" type="plane"/>
        <body name="torso" pos="0 0 0.75">
            <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
            <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
            <body name="front_left_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
                <body name="aux_1" pos="0.2 0.2 0">
                    <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
                    <body pos="0.2 0.2 0">
                        <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
            <body name="front_right_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
                <body name="aux_2" pos="-0.2 0.2 0">
                    <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
                    <body pos="-0.2 0.2 0">
                        <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
            <body name="back_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
                <body name="aux_3" pos="-0.2 -0.2 0">
                    <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
                    <body pos="-0.2 -0.2 0">
                        <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
            <body name="right_back_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
                <body name="aux_4" pos="0.2 -0.2 0">
                    <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
                    <body pos="0.2 -0.2 0">
                        <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
        </body>
        <!--By default, this goal will be hidden under the floor-->
        <site name="goal" rgba="0 1 1 0.5" pos="0. 0. -1" size="0.5"/>
        <site name="origin" rgba="1 0 0 0.5" pos="0. 0. 0." size="0.1"/>
    </worldbody>
    <actuator>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
    </actuator>
</mujoco>
"""

# class Serializable(object):

#     def __init__(self, *args, **kwargs):
#         self.__args = args
#         self.__kwargs = kwargs

#     def quick_init(self, locals_):
#         if getattr(self, "_serializable_initialized", False):
#             return
#         if sys.version_info >= (3, 0):
#             spec = inspect.getfullargspec(self.__init__)
#             # Exclude the first "self" parameter
#             if spec.varkw:
#                 kwargs = locals_[spec.varkw].copy()
#             else:
#                 kwargs = dict()
#             if spec.kwonlyargs:
#                 for key in spec.kwonlyargs:
#                     kwargs[key] = locals_[key]
#         else:
#             spec = inspect.getargspec(self.__init__)
#             if spec.keywords:
#                 kwargs = locals_[spec.keywords]
#             else:
#                 kwargs = dict()
#         if spec.varargs:
#             varargs = locals_[spec.varargs]
#         else:
#             varargs = tuple()
#         try:
#             in_order_args = [locals_[arg] for arg in spec.args][1:]
#         except KeyError:
#             in_order_args = []
#         self.__args = tuple(in_order_args) + varargs
#         self.__kwargs = kwargs
#         setattr(self, "_serializable_initialized", True)

#     def __getstate__(self):
#         return {"__args": self.__args, "__kwargs": self.__kwargs}

#     def __setstate__(self, d):
#         # convert all __args to keyword-based arguments
#         if sys.version_info >= (3, 0):
#             spec = inspect.getfullargspec(self.__init__)
#         else:
#             spec = inspect.getargspec(self.__init__)
#         in_order_args = spec.args[1:]
#         out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
#         self.__dict__.update(out.__dict__)

#     @classmethod
#     def clone(cls, obj, **kwargs):
#         assert isinstance(obj, Serializable)
#         d = obj.__getstate__()
#         d["__kwargs"] = dict(d["__kwargs"], **kwargs)
#         out = type(obj).__new__(type(obj))
#         out.__setstate__(d)
#         return out

# ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


# class MujocoEnv(mujoco_env.MujocoEnv, Serializable):
#     """
#     My own wrapper around MujocoEnv.

#     The caller needs to declare
#     """

#     def __init__(
#             self,
#             model_path,
#             frame_skip=1,
#             model_path_is_local=True,
#             automatically_set_obs_and_action_space=False,
#     ):
#         if model_path_is_local:
#             model_path = get_asset_xml(model_path)
#         if automatically_set_obs_and_action_space:
#             mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)
#         else:
#             """
#             Code below is copy/pasted from MujocoEnv's __init__ function.
#             """
#             if model_path.startswith("/"):
#                 fullpath = model_path
#             else:
#                 fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
#             if not os.path.exists(fullpath):
#                 raise IOError("File %s does not exist" % fullpath)
#             self.frame_skip = frame_skip
#             self.model = mujoco_py.MjModel(fullpath)
#             self.data = self.model.data
#             self.viewer = None

#             self.metadata = {
#                 'render.modes': ['human', 'rgb_array'],
#                 'video.frames_per_second': int(np.round(1.0 / self.dt))
#             }

#             self.init_qpos = self.model.data.qpos.ravel().copy()
#             self.init_qvel = self.model.data.qvel.ravel().copy()
#             self._seed()

#     def init_serialization(self, locals):
#         Serializable.quick_init(self, locals)

#     def log_diagnostics(self, paths):
#         pass



# def get_asset_xml(xml_name):
#     #return os.path.join(ENV_ASSET_DIR, xml_name)
#     return ant_xml

# class AntEnv(MujocoEnv):
#     def __init__(self, use_low_gear_ratio=False):
#         self.init_serialization(locals())
#         if use_low_gear_ratio:
#             xml_path = 'low_gear_ratio_ant.xml'
#         else:
#             xml_path = 'ant.xml'
#         super().__init__(
#             xml_path,
#             frame_skip=5,
#             automatically_set_obs_and_action_space=True,
#         )

#     def step(self, a):
#         torso_xyz_before = self.get_body_com("torso")
#         self.do_simulation(a, self.frame_skip)
#         torso_xyz_after = self.get_body_com("torso")
#         torso_velocity = torso_xyz_after - torso_xyz_before
#         forward_reward = torso_velocity[0] / self.dt
#         ctrl_cost = 0.  # .5 * np.square(a).sum()
#         contact_cost = 0.5 * 1e-3 * np.sum(
#             np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
#         survive_reward = 0.  # 1.0
#         reward = forward_reward - ctrl_cost - contact_cost + survive_reward
#         state = self.state_vector()
#         notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
#         done = not notdone
#         ob = self._get_obs()
#         return ob, reward, done, dict(
#             reward_forward=forward_reward,
#             reward_ctrl=-ctrl_cost,
#             reward_contact=-contact_cost,
#             reward_survive=survive_reward,
#             torso_velocity=torso_velocity,
#         )

#     def _get_obs(self):
#         # this is gym ant obs, should use rllab?
#         # if position is needed, override this in subclasses
#         return np.concatenate([
#             self.sim.data.qpos.flat[2:],
#             self.sim.data.qvel.flat,
#         ])

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
#         qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.distance = self.model.stat.extent * 0.5

#     def reset_task(self, task):
#         if task is None:
#             task = self.sample_tasks(1)[0]
#         self.set_task(task)

#     @staticmethod
#     def visualise_behaviour(env,
#                             args,
#                             policy,
#                             iter_idx,
#                             encoder=None,
#                             image_folder=None,
#                             return_pos=False,
#                             **kwargs,
#                             ):

#         num_episodes = args.max_rollouts_per_task
#         unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

#         # --- initialise things we want to keep track of ---

#         episode_prev_obs = [[] for _ in range(num_episodes)]
#         episode_next_obs = [[] for _ in range(num_episodes)]
#         episode_actions = [[] for _ in range(num_episodes)]
#         episode_rewards = [[] for _ in range(num_episodes)]

#         episode_returns = []
#         episode_lengths = []

#         if encoder is not None:
#             episode_latent_samples = [[] for _ in range(num_episodes)]
#             episode_latent_means = [[] for _ in range(num_episodes)]
#             episode_latent_logvars = [[] for _ in range(num_episodes)]
#         else:
#             episode_latent_samples = episode_latent_means = episode_latent_logvars = None

#         # --- roll out policy ---

#         # (re)set environment
#         env.reset_task()
#         state, belief, task = utl.reset_env(env, args)
#         start_obs_raw = state.clone()
#         task = task.view(-1) if task is not None else None

#         # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
#         if hasattr(args, 'hidden_size'):
#             hidden_state = torch.zeros((1, args.hidden_size)).to(device)
#         else:
#             hidden_state = None

#         # keep track of what task we're in and the position of the cheetah
#         pos = [[] for _ in range(args.max_rollouts_per_task)]
#         start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

#         for episode_idx in range(num_episodes):

#             curr_rollout_rew = []
#             pos[episode_idx].append(start_pos)

#             if episode_idx == 0:
#                 if encoder is not None:
#                     # reset to prior
#                     curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
#                     curr_latent_sample = curr_latent_sample[0].to(device)
#                     curr_latent_mean = curr_latent_mean[0].to(device)
#                     curr_latent_logvar = curr_latent_logvar[0].to(device)
#                 else:
#                     curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

#             if encoder is not None:
#                 episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
#                 episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
#                 episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

#             for step_idx in range(1, env._max_episode_steps + 1):

#                 if step_idx == 1:
#                     episode_prev_obs[episode_idx].append(start_obs_raw.clone())
#                 else:
#                     episode_prev_obs[episode_idx].append(state.clone())
#                 # act
#                 latent = utl.get_latent_for_policy(args,
#                                                    latent_sample=curr_latent_sample,
#                                                    latent_mean=curr_latent_mean,
#                                                    latent_logvar=curr_latent_logvar)
#                 _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task,
#                                           deterministic=True)

#                 (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
#                 state = state.float().reshape((1, -1)).to(device)
#                 task = task.view(-1) if task is not None else None

#                 # keep track of position
#                 pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

#                 if encoder is not None:
#                     # update task embedding
#                     curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
#                         action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
#                         hidden_state, return_prior=False)

#                     episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
#                     episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
#                     episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

#                 episode_next_obs[episode_idx].append(state.clone())
#                 episode_rewards[episode_idx].append(rew.clone())
#                 episode_actions[episode_idx].append(action.clone())

#                 if info[0]['done_mdp'] and not done:
#                     start_obs_raw = info[0]['start_state']
#                     start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
#                     start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
#                     break

#             episode_returns.append(sum(curr_rollout_rew))
#             episode_lengths.append(step_idx)

#         # clean up
#         if encoder is not None:
#             episode_latent_means = [torch.stack(e) for e in episode_latent_means]
#             episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

#         episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
#         episode_next_obs = [torch.cat(e) for e in episode_next_obs]
#         episode_actions = [torch.stack(e) for e in episode_actions]
#         episode_rewards = [torch.cat(e) for e in episode_rewards]

#         # plot the movement of the ant
#         # print(pos)
#         plt.figure(figsize=(5, 4 * num_episodes))
#         min_dim = -3.5
#         max_dim = 3.5
#         span = max_dim - min_dim

#         for i in range(num_episodes):
#             plt.subplot(num_episodes, 1, i + 1)

#             x = list(map(lambda p: p[0], pos[i]))
#             y = list(map(lambda p: p[1], pos[i]))
#             plt.plot(x[0], y[0], 'bo')

#             plt.scatter(x, y, 1, 'g')

#             curr_task = env.get_task()
#             plt.title('task: {}'.format(curr_task), fontsize=15)
#             if 'Goal' in args.env_name:
#                 plt.plot(curr_task[0], curr_task[1], 'rx')

#             plt.ylabel('y-position (ep {})'.format(i), fontsize=15)

#             if i == num_episodes - 1:
#                 plt.xlabel('x-position', fontsize=15)
#                 plt.ylabel('y-position (ep {})'.format(i), fontsize=15)
#             plt.xlim(min_dim - 0.05 * span, max_dim + 0.05 * span)
#             plt.ylim(min_dim - 0.05 * span, max_dim + 0.05 * span)

#         plt.tight_layout()
#         if image_folder is not None:
#             plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
#             plt.close()
#         else:
#             plt.show()

#         if not return_pos:
#             return episode_latent_means, episode_latent_logvars, \
#                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
#                    episode_returns
#         else:
#             return episode_latent_means, episode_latent_logvars, \
#                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
#                    episode_returns, pos

from gymnasium.envs.mujoco.ant_v5 import AntEnv

class AntDirEnv(AntEnv):
    """
    Forward/backward ant direction environment
    """

    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(AntDirEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self.goal_direction), np.sin(self.goal_direction))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            task=self.get_task()
        )

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return [random.choice([-1.0, 1.0]) for _ in range(n_tasks, )]

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_direction = task

    def get_task(self):
        return np.array([self.goal_direction])


class AntDir2DEnv(AntDirEnv):
    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        directions = np.array([random.gauss(mu=0, sigma=1) for _ in range(n_tasks * 2)]).reshape((n_tasks, 2))
        directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        return directions


class AntDirOracleEnv(AntDirEnv):
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            [self.goal_direction],
        ])


class AntDir2DOracleEnv(AntDir2DEnv):
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            [self.goal_direction],
        ])


def eval(
    agent,
    config,
    num_eval_episodes: int = 5,
    max_episode_steps: int = 200,
    step_t: int = 0,
):
    """
    Evaluate a trained VariBAD agent on AntDirEnv. Runs num_eval_episodes per task.
    Passes x_t (prev_obs, prev_action, prev_reward, obs, done) to agent.act so the
    VAE encoder runs and the policy receives the inferred latent z (same as in training).
    Returns dict with mean/std return per task.
    """
    eval_env = AntDirEnv(max_episode_steps=max_episode_steps)
    #agent.ppo.eval()
    #agent.vae.eval()
    device = config.device
    returns = np.zeros((num_eval_episodes,), dtype=np.float32)
    results = {}

    for ep in range(num_eval_episodes):
        agent.reset()
        obs, _ = eval_env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        step = 0
        done = False
        rewards = []
        prev_obs, prev_action, prev_reward = None, None, 0.0

        while step < max_episode_steps and not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                if prev_obs is not None:
                    x_t_np = np.concatenate([
                        prev_obs.flatten(),
                        np.asarray(prev_action, dtype=np.float32).flatten(),
                        [prev_reward],
                        obs.flatten(),
                        [0.0],
                    ], dtype=np.float32)
                    x_t = torch.from_numpy(x_t_np).unsqueeze(0).to(device)
                else:
                    x_t = None
                action, _, _ = agent.act(obs_t, x_t, values=False)
                action_np = action.cpu().numpy()[0]

            next_obs, reward, done, info = eval_env.step(action_np)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            rewards.append(reward)

            prev_obs, prev_action, prev_reward = obs, action_np, reward
            obs = next_obs
            step += 1

        returns[ep] = np.sum(rewards)

    results[f"reward_mean"] = float(returns.mean())
    results[f"reward_std"] = float(returns.std())

    if config.use_wandb:
        wandb.log({f"eval/{k}": v for k, v in results.items()}, step=step_t)
    eval_env.close()
    return results


# =============================================================================
# VariBAD training on this Ant env (no continual wrapper)
# =============================================================================

if __name__ == "__main__":

    config = VaribadConfig(
        discrete=False,
        env=EnvConfig(domain_name="ant_dir", max_episode_steps=400),
        num_episodes=1000,
        batch_size=3200,
        minibatch_size=4,
        ppo=PPOConfig(batch_size=3200, minibatch_size=4),
        seed=0,
        device="cuda",
        z_dim=5,
        beta_kl=0.1,
        vae_lr=0.001,
        use_wandb=True,
        proj_name="rl-algorithms",
        algo_name="variBAD",
    )

    wandb.init(
        project=config.proj_name,
        group=f"{config.env.domain_name}-s{config.env.seed}",
        name=f"{config.algo_name}-s{config.seed}",
        config=asdict(config)
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    env = AntDirEnv(max_episode_steps=400)
    s_dim, a_dim, discrete = get_env_dims(env)
    config.s_dim = s_dim
    config.a_dim = a_dim
    config.discrete = discrete

    print(f"s_dim: {s_dim}, a_dim: {a_dim}, discrete: {discrete}")

    agent = VariBAD(config)
    replay_buffer = SimpleTrajBuffer(device=config.device, capacity_episodes=50_000)

    obs_buffer = torch.zeros((config.ppo.batch_size + 1, s_dim + config.z_dim)).to(config.device)
    actions_buffer = torch.zeros((config.ppo.batch_size + 1, a_dim)).to(config.device)
    logprobs_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)
    rewards_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)
    dones_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)
    values_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)

    ppo_buffer_idx = 0
    for episode in tqdm(range(config.num_episodes), desc="Train"):
        # Sample a new task each episode (use the env's own API)
        task = env.sample_tasks(1)[0]
        env.set_task(task)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float32)

        vae_episode = {k: [] for k in ["s", "a", "r", "sp", "d"]}
        prev_obs, prev_action, prev_reward = None, None, 0.0
        step = 0

        while step < config.env.max_episode_steps:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=config.device, dtype=torch.float32).unsqueeze(0)
                if prev_obs is not None:
                    x_t_np = np.concatenate([
                        prev_obs.flatten(),
                        np.asarray(prev_action, dtype=np.float32).flatten(),
                        [prev_reward],
                        obs.flatten(),
                        [0.0],
                    ], dtype=np.float32)
                    x_t = torch.from_numpy(x_t_np).unsqueeze(0).to(config.device)
                else:
                    x_t = None
                action, logprob, entropy, value, info_dict = agent.act(obs_t, x_t, values=True)
                z_pol = info_dict["z_pol"]
                action_np = action.cpu().numpy()[0]

            next_obs, reward, done, info = env.step(action_np)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            # Old gym: done is bool; treat as terminated, no truncation
            terminated, truncated = bool(done), False
            done = terminated or truncated

            obs_aug = torch.cat([obs_t.squeeze(0), z_pol.squeeze(0)], dim=-1)
            obs_buffer[ppo_buffer_idx] = obs_aug
            actions_buffer[ppo_buffer_idx] = action.squeeze(0)
            logprobs_buffer[ppo_buffer_idx] = logprob.item()
            rewards_buffer[ppo_buffer_idx] = reward
            dones_buffer[ppo_buffer_idx] = float(done)
            values_buffer[ppo_buffer_idx] = value.item()
            ppo_buffer_idx += 1

            vae_episode["s"].append(obs)
            vae_episode["a"].append(action_np)
            vae_episode["r"].append([reward])
            vae_episode["sp"].append(next_obs)
            vae_episode["d"].append([float(done)])

            prev_obs, prev_action, prev_reward = obs, action_np, reward
            obs = next_obs
            step += 1

            if ppo_buffer_idx == config.ppo.batch_size + 1:
                rollout = {
                    "obs": obs_buffer[:config.ppo.batch_size]   ,
                    "actions": actions_buffer[:config.ppo.batch_size],
                    "logprobs": logprobs_buffer[:config.ppo.batch_size],
                    "rewards": rewards_buffer[:config.ppo.batch_size],
                    "dones": dones_buffer[:config.ppo.batch_size],
                    "values": values_buffer[:config.ppo.batch_size + 1],
                }
                agent.update(rollout, replay_buffer)
                ppo_buffer_idx = 0

            if done:
                break

        for key in vae_episode:
            vae_episode[key] = torch.tensor(np.array(vae_episode[key]), dtype=torch.float32)
        replay_buffer.add_episode(vae_episode)

        eval(agent, config, num_eval_episodes=5, max_episode_steps=config.env.max_episode_steps, step_t=episode)




