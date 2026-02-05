import numpy as np
import torch


class RolloutStorageVAE(object):
    def __init__(self, num_processes, max_trajectory_len, zero_pad, max_num_rollouts,
                 state_dim, action_dim, vae_buffer_add_thresh, task_dim):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """

        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.task_dim = task_dim

        self.vae_buffer_add_thresh = vae_buffer_add_thresh  # prob of adding new trajectories
        self.max_buffer_size = max_num_rollouts  # maximum buffer len (number of trajectories)
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.buffer_len = 0  # how much of the buffer has been filled

        # how long a trajectory can be at max (horizon)
        self.max_traj_len = max_trajectory_len
        # whether to zero-pad to maximum length (zero's at the end!)
        self.zero_pad = zero_pad

        # buffers for completed rollouts (stored on CPU)
        if self.max_buffer_size > 0:
            self.prev_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.next_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
            self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, action_dim))
            self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
            if task_dim is not None:
                self.tasks = torch.zeros((self.max_buffer_size, task_dim))
            else:
                self.tasks = None
            self.trajectory_lens = [0] * self.max_buffer_size

        # storage for each running process (stored on GPU)
        self.num_processes = num_processes
        self.curr_timestep = torch.zeros((num_processes)).long()  # count environment steps so we know where to insert
        self.running_prev_state = torch.zeros((self.max_traj_len, num_processes, state_dim)).to(self.args.device)  # for each episode will have obs 0...N-1
        self.running_next_state = torch.zeros((self.max_traj_len, num_processes, state_dim)).to(self.args.device)  # for each episode will have obs 1...N
        self.running_rewards = torch.zeros((self.max_traj_len, num_processes, 1)).to(self.args.device)
        self.running_actions = torch.zeros((self.max_traj_len, num_processes, action_dim)).to(self.args.device)
        if task_dim is not None:
            self.running_tasks = torch.zeros((num_processes, task_dim)).to(self.args.device)
        else:
            self.running_tasks = None

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep

    def insert(self, prev_state, actions, next_state, rewards, done, task):

        # add to temporary buffer

        already_inserted = False
        if len(np.unique(self.curr_timestep)) == 1:
            self.running_prev_state[self.curr_timestep[0]] = prev_state
            self.running_next_state[self.curr_timestep[0]] = next_state
            self.running_rewards[self.curr_timestep[0]] = rewards
            self.running_actions[self.curr_timestep[0]] = actions
            if task is not None:
                self.running_tasks = task
            self.curr_timestep += 1
            already_inserted = True

        already_reset = False
        if done.sum() == self.num_processes:  # check if we can process the entire batch at once

            # add to permanent (up to max_buffer_len) buffer
            if self.max_buffer_size > 0:
                if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                    # check where to insert data
                    if self.insert_idx + self.num_processes > self.max_buffer_size:
                        # keep track of how much we filled the buffer (for sampling from it)
                        self.buffer_len = self.insert_idx
                        # this will keep some entries at the end of the buffer without overwriting them,
                        # but the buffer is large enough to make this negligible
                        self.insert_idx = 0
                    else:
                        self.buffer_len = max(self.buffer_len, self.insert_idx)
                    # add; note: num trajectories are along dim=1,
                    # trajectory length along dim=0, to match pytorch RNN interface
                    self.prev_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_prev_state
                    self.next_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_next_state
                    self.actions[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_actions
                    self.rewards[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_rewards
                    if (self.tasks is not None) and (self.running_tasks is not None):
                        insert_shape = self.tasks[self.insert_idx:self.insert_idx+self.num_processes].shape
                        self.tasks[self.insert_idx:self.insert_idx+self.num_processes] = self.running_tasks.reshape(insert_shape)
                    self.trajectory_lens[self.insert_idx:self.insert_idx+self.num_processes] = self.curr_timestep.clone()
                    self.insert_idx += self.num_processes

            # empty running buffer
            self.running_prev_state *= 0
            self.running_next_state *= 0
            self.running_rewards *= 0
            self.running_actions *= 0
            if self.running_tasks is not None:
                self.running_tasks *= 0
            self.curr_timestep *= 0

            already_reset = True

        if (not already_inserted) or (not already_reset):

            for i in range(self.num_processes):

                if not already_inserted:
                    self.running_prev_state[self.curr_timestep[i], i] = prev_state[i]
                    self.running_next_state[self.curr_timestep[i], i] = next_state[i]
                    self.running_rewards[self.curr_timestep[i], i] = rewards[i]
                    self.running_actions[self.curr_timestep[i], i] = actions[i]
                    if self.running_tasks is not None:
                        self.running_tasks[i] = task[i]
                    self.curr_timestep[i] += 1

                if not already_reset:
                    # if we are at the end of a task, dump the data into the larger buffer
                    if done[i]:

                        # add to permanent (up to max_buffer_len) buffer
                        if self.max_buffer_size > 0:
                            if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                                # check where to insert data
                                if self.insert_idx + 1 > self.max_buffer_size:
                                    # keep track of how much we filled the buffer (for sampling from it)
                                    self.buffer_len = self.insert_idx
                                    # this will keep some entries at the end of the buffer without overwriting them,
                                    # but the buffer is large enough to make this negligible
                                    self.insert_idx = 0
                                else:
                                    self.buffer_len = max(self.buffer_len, self.insert_idx)
                                # add; note: num trajectories are along dim=1,
                                # trajectory length along dim=0, to match pytorch RNN interface
                                self.prev_state[:, self.insert_idx] = self.running_prev_state[:, i].to('cpu')
                                self.next_state[:, self.insert_idx] = self.running_next_state[:, i].to('cpu')
                                self.actions[:, self.insert_idx] = self.running_actions[:, i].to('cpu')
                                self.rewards[:, self.insert_idx] = self.running_rewards[:, i].to('cpu')
                                if self.tasks is not None:
                                    self.tasks[self.insert_idx] = self.running_tasks[i].to('cpu')
                                self.trajectory_lens[self.insert_idx] = self.curr_timestep[i].clone()
                                self.insert_idx += 1

                        # empty running buffer
                        self.running_prev_state[:, i] *= 0
                        self.running_next_state[:, i] *= 0
                        self.running_rewards[:, i] *= 0
                        self.running_actions[:, i] *= 0
                        if self.running_tasks is not None:
                            self.running_tasks[i] *= 0
                        self.curr_timestep[i] = 0

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        # TODO: check if we can get rid of num_enc_len and num_rollouts (call it batchsize instead)

        batchsize = min(self.buffer_len, batchsize)

        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]

        # select the rollouts we want
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        if self.tasks is not None:
            tasks = self.tasks[rollout_indices].to(self.args.device)
        else:
            tasks = None

        return prev_obs.to(self.args.device), next_obs.to(self.args.device), actions.to(self.args.device), \
               rewards.to(self.args.device), tasks, trajectory_lens


"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])


class OnlineStorage(object):
    def __init__(self,
                 args, num_steps, num_processes,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards):

        self.args = args
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.task_dim = task_dim

        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, num_processes, state_dim)
        if self.args.pass_latent_to_policy:
            # latent variables (of VAE)
            self.latent_dim = latent_dim
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            # hidden states of RNN (necessary if we want to re-compute embeddings)
            self.hidden_size = hidden_size
            self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)
            # next_state will include s_N when state was reset, skipping s_0
            # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
            self.next_state = torch.zeros(num_steps, num_processes, state_dim)
        else:
            self.latent_mean = None
            self.latent_logvar = None
            self.latent_samples = None
        if self.args.pass_belief_to_policy:
            self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim)
        else:
            self.beliefs = None
        if self.args.pass_task_to_policy:
            self.tasks = torch.zeros(num_steps + 1, num_processes, task_dim)
        else:
            self.tasks = None

        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.to_device()

    def to_device(self):
        if self.args.pass_state_to_policy:
            self.prev_state = self.prev_state.to(self.args.device)
        if self.args.pass_latent_to_policy:
            self.latent_samples = [t.to(self.args.device) for t in self.latent_samples]
            self.latent_mean = [t.to(self.args.device) for t in self.latent_mean]
            self.latent_logvar = [t.to(self.args.device) for t in self.latent_logvar]
            self.hidden_states = self.hidden_states.to(self.args.device)
            self.next_state = self.next_state.to(self.args.device)
        if self.args.pass_belief_to_policy:
            self.beliefs = self.beliefs.to(self.args.device)
        if self.args.pass_task_to_policy:
            self.tasks = self.tasks.to(self.args.device)
        self.rewards_raw = self.rewards_raw.to(self.args.device)
        self.rewards_normalised = self.rewards_normalised.to(self.args.device)
        self.done = self.done.to(self.args.device)
        self.masks = self.masks.to(self.args.device)
        self.bad_masks = self.bad_masks.to(self.args.device)
        self.value_preds = self.value_preds.to(self.args.device)
        self.returns = self.returns.to(self.args.device)
        self.actions = self.actions.to(self.args.device)

    def insert(self,
               state,
               belief,
               task,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               bad_masks,
               done,
               #
               hidden_states=None,
               latent_sample=None,
               latent_mean=None,
               latent_logvar=None,
               ):
        self.prev_state[self.step + 1].copy_(state)
        if self.args.pass_belief_to_policy:
            self.beliefs[self.step + 1].copy_(belief)
        if self.args.pass_task_to_policy:
            self.tasks[self.step + 1].copy_(task)
        if self.args.pass_latent_to_policy:
            self.latent_samples.append(latent_sample.detach().clone())
            self.latent_mean.append(latent_mean.detach().clone())
            self.latent_logvar.append(latent_logvar.detach().clone())
            self.hidden_states[self.step + 1].copy_(hidden_states.detach())
        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.done[self.step + 1].copy_(done)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.prev_state[0].copy_(self.prev_state[-1])
        if self.args.pass_belief_to_policy:
            self.beliefs[0].copy_(self.beliefs[-1])
        if self.args.pass_task_to_policy:
            self.tasks[0].copy_(self.tasks[-1])
        if self.args.pass_latent_to_policy:
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            self.hidden_states[0].copy_(self.hidden_states[-1])
        self.done[0].copy_(self.done[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.action_log_probs = None

    def compute_returns(self, next_value, use_gae, gamma, tau, use_proper_time_limits=True):

        if self.normalise_rewards:
            rewards = self.rewards_normalised.clone()
        else:
            rewards = self.rewards_raw.clone()

        self._compute_returns(next_value=next_value, rewards=rewards, value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=gamma, tau=tau, use_gae=use_gae, use_proper_time_limits=use_proper_time_limits)

    def _compute_returns(self, next_value, rewards, value_preds, returns, gamma, tau, use_gae, use_proper_time_limits):

        if use_proper_time_limits:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[
                        step + 1] + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        return len(self.prev_state) * self.num_processes

    def before_update(self, policy):
        latent = get_latent_for_policy(self.args,
                                           latent_sample=torch.stack(
                                               self.latent_samples[:-1]) if self.latent_samples is not None else None,
                                           latent_mean=torch.stack(
                                               self.latent_mean[:-1]) if self.latent_mean is not None else None,
                                           latent_logvar=torch.stack(
                                               self.latent_logvar[:-1]) if self.latent_mean is not None else None)
        _, action_log_probs, _ = policy.evaluate_actions(self.prev_state[:-1],
                                                         latent,
                                                         self.beliefs[:-1] if self.beliefs is not None else None,
                                                         self.tasks[:-1] if self.tasks is not None else None,
                                                         self.actions)
        self.action_log_probs = action_log_probs.detach()

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if self.args.pass_state_to_policy:
                state_batch = self.prev_state[:-1].reshape(-1, *self.prev_state.size()[2:])[indices]
            else:
                state_batch = None
            if self.args.pass_latent_to_policy:
                latent_sample_batch = torch.cat(self.latent_samples[:-1])[indices]
                latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
                latent_logvar_batch = torch.cat(self.latent_logvar[:-1])[indices]
            else:
                latent_sample_batch = latent_mean_batch = latent_logvar_batch = None
            if self.args.pass_belief_to_policy:
                belief_batch = self.beliefs[:-1].reshape(-1, *self.beliefs.size()[2:])[indices]
            else:
                belief_batch = None
            if self.args.pass_task_to_policy:
                task_batch = self.tasks[:-1].reshape(-1, *self.tasks.size()[2:])[indices]
            else:
                task_batch = None

            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield state_batch, belief_batch, task_batch, \
                  actions_batch, \
                  latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):

    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None

    if args.add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if args.sample_embeddings:
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1:
        latent = latent.squeeze(0)

    return latent