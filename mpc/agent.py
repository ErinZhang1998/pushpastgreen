import numpy as np
import copy
import pickle 
import cv2 
import os 


class Bad_Traj_Exception(Exception):
    def __init__(self):
        pass


class Image_Exception(Exception):
    def __init__(self):
        pass


class Environment_Exception(Exception):
    def __init__(self):
        pass

class GeneralAgent(object):

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.T = self._hyperparams['T']
        self.data_root = self._hyperparams['data_root']
        self._setup_world(0)

    def _setup_world(self, itr):
        env_type, env_params = self._hyperparams['env']
        self.env = env_type(env_params, self._reset_state)

        # self._hyperparams['adim'] = self.adim = self.env.adim
        # self._hyperparams['sdim'] = self.sdim = self.env.sdim
        # self._hyperparams['ncam'] = self.ncam = self.env.ncam

    def sample(self, policy, i_traj):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """

        traj_ok, obs_dict, policy_outs, agent_data = False, None, None, None
        i_trial = 0

        while not traj_ok and i_trial < self._hyperparams.get('imax', 10):
            i_trial += 1
            try:
                agent_data, obs_dict, policy_outs = self.rollout(policy, i_trial, i_traj)
                traj_ok = agent_data['traj_ok']
            except (Image_Exception, Environment_Exception) as rolloutexp:
                traj_ok = False

        if not traj_ok:
            raise Bad_Traj_Exception

        print('needed {} trials'.format(i_trial))

        return agent_data, obs_dict, policy_outs

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        raise 
        return obs

    def rollout(self, policy, i_trial, i_traj):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_trial: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        agent_data, policy_outputs = {}, []

        # Take the sample.
        t = 0
        done = self._hyperparams['T'] <= 0
        self.env.reset()
        policy.reset()

        while not done:
            pi_t = policy.act(t, i_traj, state=None)
            policy_outputs.append(pi_t)

            returned_infos = self.env.step(copy.deepcopy(pi_t['actions']))  #self._post_process_obs(, agent_data)

            if 'rejection_sample' in self._hyperparams and 'rejection_end_early' in self._hyperparams:
                print('traj rejected!')
                if self._hyperparams['rejection_sample'] > i_trial and not self.env.goal_reached():
                    return {'traj_ok': False}, None, None
            
            self.save_action_info(returned_infos)

            if self.env.goal_reached():
                done = True
            if (self._hyperparams['T'] - 1) == t:
                done = True
            t += 1

        traj_ok = self.env.valid_rollout()
        if 'rejection_sample' in self._hyperparams:
            if self._hyperparams['rejection_sample'] > i_trial:
                traj_ok = self.env.goal_reached()

        # self._required_rollout_metadata(agent_data, traj_ok, t, i_traj, i_trial, reset_state)
        return agent_data, None, policy_outputs
    
    
            
    @property
    def record_path(self):
        return self._hyperparams['data_save_dir'] + '/record/'
