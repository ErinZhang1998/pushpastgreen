import numpy as np
import scipy
import copy
import sys


def make_blockdiagonal(cov, nactions, adim):
    mat = np.zeros_like(cov)
    for i in range(nactions-1):
        mat[i*adim:i*adim + adim*2, i*adim:i*adim + adim*2] = np.ones([adim*2, adim*2])
    newcov = cov*mat
    return newcov

def construct_initial_mean_sigma(hp):
    diag = []
    means = []
    for var_name, var_dict in hp['initial_sampling_configurations'].items():
        diag.append(var_dict['std']**2)
        means.append(var_dict['mean'])

    diag = np.tile(diag, hp['nactions'])
    diag = np.array(diag)
    sigma = np.diag(diag)
    mean = np.tile(means, hp['nactions'])
    return mean, sigma

# def repeat_parameters_for_truncated_normal

def construct_initial_mean_sigma_for_truncated_normal(hp, nsamples):
    std = []
    means = []
    bounds = []
    for var_name, var_dict in hp['initial_sampling_configurations'].items():
        std.append(var_dict['std'])
        means.append(var_dict['mean'])
        bounds.append([var_dict['min'], var_dict['max']])

    sigma = np.repeat(np.array(std).reshape(1,-1), nsamples * hp['nactions'], axis=0).reshape(-1,)
    mean = np.repeat(np.array(means).reshape(1,-1), nsamples * hp['nactions'], axis=0).reshape(-1,)
    bounds_arr = np.repeat(np.array(bounds)[np.newaxis,...], nsamples * hp['nactions'], axis=0).reshape((-1,2))
    truncnorm_a = (bounds_arr[:, 0] - mean) / sigma
    truncnorm_b = (bounds_arr[:, 1] - mean) / sigma
    return truncnorm_a, truncnorm_b, mean, sigma

def discretize(actions, hp):
    """
    discretize and clip between 0 and 4
    :param actions:
    :return:
    """
    for var_i, (var_name, var_dict) in enumerate(hp['initial_sampling_configurations'].items()):
        max_discrete_ind = int((var_dict['max'] - var_dict['min']) / var_dict['discrete'])
        print(f":: {var_name}: max_discrete_ind={max_discrete_ind}")
        # if var_name in ['x','y']:
        #     max_discrete_ind -= 1
        if var_name == 'theta':
            if np.isclose(var_dict['max'], 2*np.pi, atol=np.radians(10), rtol=0.0):
                max_discrete_ind -= 1
        discretes = np.clip(np.floor( (actions[:, :, var_i] - var_dict['min']) / var_dict['discrete'] ), 0, max_discrete_ind)
        if var_name in ['x','y']:
            # discretes += 0.5
            actions[:, :, var_i] = discretes 
        else:
            actions[:, :, var_i] = discretes * var_dict['discrete'] + var_dict['min']
        
    return actions

class TruncatedGaussianCEMSampler(object):
    def __init__(self, hp):
        self._hp = hp
        self._adim = len(hp['initial_sampling_configurations'])
        self._chosen_actions = []
        self._best_action_plans = []
        self._sigma, self._sigma_prev = None, None
        self._mean = None
        self._last_reduce = False
        assert len(self._hp['var_order']) == len(hp['initial_sampling_configurations'])

        bounds = []
        for var_name, var_dict in self._hp['initial_sampling_configurations'].items():
            bounds.append([var_dict['min'], var_dict['max']])
        self.bounds = np.array(bounds)

    def _valid(self, actions):
        # for var_i, (var_name, var_dict) in enumerate(self._hp['initial_sampling_configurations'].items()):
        #     if var_dict['invalid_range'] is None:
        #         continue 
        #     actions[..., var_i] >= 
        if self._hp['invalid_range'] is None:
            return np.ones_like(actions[..., 0]).astype(bool)

        x_idx = self._hp['var_order'].index('x')
        y_idx = self._hp['var_order'].index('y')
        x_invalid = np.logical_and(actions[..., x_idx] >= self._hp['invalid_range'][0], actions[..., x_idx] <= self._hp['invalid_range'][1])
        y_invalid = np.logical_and(actions[..., y_idx] >= self._hp['invalid_range'][2], actions[..., y_idx] <= self._hp['invalid_range'][3])
        invalid_actions = np.logical_and(x_invalid, y_invalid)
        return np.logical_not(invalid_actions)
    
    def _discretize(self, actions):
        for var_i, (var_name, var_dict) in enumerate(self._hp['initial_sampling_configurations'].items()):
            
            max_discrete_ind = int((var_dict['max'] - var_dict['min']) / var_dict['discrete'])
            # print(f":: {var_name}: max_discrete_ind={max_discrete_ind}")
            if var_name == 'theta':
                if np.isclose(var_dict['max'], 2*np.pi, atol=np.radians(10), rtol=0.0):
                    max_discrete_ind -= 1
                
            discretes = np.floor( (actions[..., var_i] - var_dict['min']) / var_dict['discrete'] )

            if var_name in ['x','y']:
                # discretes += 0.5
                actions[..., var_i] = discretes 
            else:
                actions[..., var_i] = discretes * var_dict['discrete'] + var_dict['min']
        return actions
    
    def _dediscretize(self, actions):
        for var_i, (var_name, var_dict) in enumerate(self._hp['initial_sampling_configurations'].items()):
            if var_name in ['x','y']:
                actions[:, :, var_i] = actions[:, :, var_i] *  var_dict['discrete'] + var_dict['min']
        return actions
    
    def sample_initial_actions(self, t, nsamples, current_state):
        valid_actions = []
        while len(valid_actions) < nsamples:
            actions = np.zeros((nsamples, self._hp['nactions'], len(self._hp['initial_sampling_configurations'])))
            for var_i, (var_name, var_dict) in enumerate(self._hp['initial_sampling_configurations'].items()):
                # x: 28 | y: 26 | theta : 8 | z_level : 2
                max_discrete_ind = int((var_dict['max'] - var_dict['min']) / var_dict['discrete'])
                if var_name in ['x','y','z_level']:
                    max_discrete_ind += 1
                
                sampled_values = np.random.choice(max_discrete_ind, nsamples * self._hp['nactions'], replace=True).reshape(nsamples, self._hp['nactions'])
                
                if var_name in ['x','y', 'z_level']:
                    actions[:, :, var_i] = sampled_values 
                else:
                    actions[:, :, var_i] = sampled_values * var_dict['discrete'] + var_dict['min']
            valid_ones = self._valid(actions) # (n_samples, nactions)
            valid_actions_this_round = np.sum(valid_ones, axis=1) == self._hp['nactions'] # (n_samples,)
            
            valid_actions += list(actions[valid_actions_this_round])
            
        
        final_actions = np.stack(valid_actions[:nsamples])
        return final_actions
        self._truncnorm_a, self._truncnorm_b, self._mean, self._sigma = construct_initial_mean_sigma_for_truncated_normal(self._hp, nsamples)
        self._sigma_prev = self._sigma            
        return self._sample(nsamples)

    def sample_next_actions(self, n_samples, best_actions, scores):
        best_actions = self._dediscretize(best_actions)
        self._fit_gaussians(best_actions, n_samples)
        # import pdb;pdb.set_trace()
        return self._sample(n_samples)

    def _sample(self, n_samples):
        # actions = np.random.multivariate_normal(self._mean, self._sigma, n_samples)
        num_valid_actions = 0
        valid_actions = []
        while len(valid_actions) < n_samples:
            actions = scipy.stats.truncnorm.rvs(self._truncnorm_a, self._truncnorm_b, loc=self._mean, scale=self._sigma)
            actions = actions.reshape(n_samples, self._hp['nactions'], self._adim)
            actions = self._discretize(actions)
            valid_ones = self._valid(actions) # (n_samples, nactions)
            valid_actions_this_round = np.sum(valid_ones, axis=1) == self._hp['nactions'] # (n_samples,)
            
            valid_actions += list(actions[valid_actions_this_round])
            
            
        final_actions = np.stack(valid_actions[:n_samples])
        return final_actions

    def _fit_gaussians(self, actions, n_samples):
        # actions = actions.reshape(-1, self._hp['nactions'], self._hp['repeat'], self._adim)
        # actions = actions[:, :, -1, :]  # taking only one of the repeated actions
        actions_flat = actions.reshape(-1, self._hp['nactions'] * self._adim)
        std = np.std(actions_flat, axis=0) # (nactions * adim)
        
        std_reshaped = std.reshape((self._hp['nactions'], self._adim))
        std_reshaped[:,2] += np.radians(10)
        std = std_reshaped.reshape(-1,)
        
        means = np.mean(actions_flat, axis=0)
        sigma = np.repeat(np.array(std).reshape(1,-1), n_samples, axis=0).reshape(-1,)
        mean = np.repeat(np.array(means).reshape(1,-1), n_samples, axis=0).reshape(-1,)
        bounds_arr = np.repeat(self.bounds[np.newaxis,...], n_samples, axis=0).reshape((-1,2))
        self._truncnorm_a = (bounds_arr[:, 0] - mean) / sigma
        self._truncnorm_b = (bounds_arr[:, 1] - mean) / sigma

        self._sigma = sigma
        self._mean = mean
    
    def log_best_action(self, action, best_action_plans):
        """
        Some sampling distributions may change given the taken action

        :param action: action executed
        :param best_action_plans: batch of next planned actions (after this timestep) ordered in ascending cost
        :return: None
        """
        self._chosen_actions.append(action.copy())
        self._best_action_plans.append(best_action_plans)
    
    @property
    def chosen_actions(self):
        """
        :return: actions chosen by policy thus far
        """
        return np.array(self._chosen_actions)
    
    @staticmethod
    def get_default_hparams():
        hparams = {
            'invalid_range' : None,
            'var_order' : ['x','y','theta','z_level'],
            'initial_sampling_configurations' : {
                'x' : {
                    'mean' : 0.2,
                    'std' : 0.3,
                    'min' : 0,
                    'max' : 0.4,
                    'discrete' : 0.02,
                },
                'y' : {
                    'mean' : 0.2,
                    'std' : 0.3,
                    'min' : 0,
                    'max' : 0.4,
                    'discrete' : 0.02,
                },
                # 'theta' : {
                #     'mean' : np.radians(210 / 2),
                #     'std' : np.pi * 2,
                #     'min' : 0,
                #     'max' : np.pi+np.pi/6,
                #     'discrete' : np.pi / 6,
                # },
                'theta' : {
                    'mean' : np.pi,
                    'std' : np.pi,
                    'min' : 0,
                    'max' : 2*np.pi,
                    'discrete' : np.pi / 4,
                },
                'z_level' : {
                    'mean' : 1,
                    'std' : 2,
                    'min' : 0,
                    'max' : 2,
                    'discrete' : 1,
                }
            },
            'rejection_sampling': False,
            'cov_blockdiag': False,
            'smooth_cov': False,
            'nactions': 1,
            'repeat': 1, # number of times to execute the same action
        }
        return hparams

def dediscretize(actions, hp):
    for var_i, (var_name, var_dict) in enumerate(hp['initial_sampling_configurations'].items()):
        if var_name in ['x','y']:
            actions[:, :, var_i] = actions[:, :, var_i] *  var_dict['discrete'] + var_dict['min']
    return actions

class GaussianCEMSampler(object):
    def __init__(self, hp):
        self._hp = hp
        self._adim = len(hp['initial_sampling_configurations'])
        self._chosen_actions = []
        self._best_action_plans = []
        self._sigma, self._sigma_prev = None, None
        self._mean = None
        self._last_reduce = False
        assert len(self._hp['var_order']) == len(hp['initial_sampling_configurations'])

    def sample_initial_actions(self, t, nsamples, current_state):
        self._mean, self._sigma = construct_initial_mean_sigma(self._hp)
        self._sigma_prev = self._sigma            
        return self._sample(nsamples)

    def sample_next_actions(self, n_samples, best_actions, scores):
        best_actions = dediscretize(best_actions, self._hp)
        self._fit_gaussians(best_actions)
        return self._sample(n_samples)

    def _sample(self, n_samples):
        if self._hp['rejection_sampling']:
            return self._sample_actions_rej(n_samples)
        return self._sample_actions(n_samples)

    def _sample_actions(self, n_samples):
        actions = np.random.multivariate_normal(self._mean, self._sigma, n_samples)
        # s = scipy.stats.truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
        actions = actions.reshape(n_samples, self._hp['nactions'], self._adim)
        actions = discretize(actions, self._hp)
        actions = np.repeat(actions, self._hp['repeat'], axis=1)
        return actions

    def _fit_gaussians(self, actions):
        actions = actions.reshape(-1, self._hp['nactions'], self._hp['repeat'], self._adim)
        actions = actions[:, :, -1, :]  # taking only one of the repeated actions
        actions_flat = actions.reshape(-1, self._hp['nactions'] * self._adim)

        self._sigma = np.cov(actions_flat, rowvar=False, bias=False)
        # if self._hp['cov_blockdiag']:
        #     self._sigma = make_blockdiagonal(self._sigma, self._hp['nactions'], self._adim)
        # if self._hp['smooth_cov']:
        #     self._sigma = 0.5 * self._sigma + 0.5 * self._sigma_prev
        #     self._sigma_prev = self._sigma
        self._mean = np.mean(actions_flat, axis=0)
    
    def log_best_action(self, action, best_action_plans):
        """
        Some sampling distributions may change given the taken action

        :param action: action executed
        :param best_action_plans: batch of next planned actions (after this timestep) ordered in ascending cost
        :return: None
        """
        self._chosen_actions.append(action.copy())
        self._best_action_plans.append(best_action_plans)
    
    @property
    def chosen_actions(self):
        """
        :return: actions chosen by policy thus far
        """
        return np.array(self._chosen_actions)
    
    @staticmethod
    def get_default_hparams():
        hparams = {
            # 'parameter_bounds' : {
            #     'x' : {
            #         'min' : 0,
            #         'max' : 0.4,
            #         'discrete' : 0.02,
            #     },
            #     'y' : {
            #         'min' : 0,
            #         'max' : 0.4,
            #         'discrete' : 0.02,
            #     },
            #     'theta' : {
            #         'min' : 0,
            #         'max' : np.pi,
            #         'discrete' : np.pi / 6,
            #     },
            # },
            'var_order' : ['x','y','theta'],
            'initial_sampling_configurations' : {
                'x' : {
                    'mean' : 0.2,
                    'std' : 0.25,
                    'min' : 0,
                    'max' : 0.4,
                    'discrete' : 0.02,
                    # 'reject' : 2.7,
                },
                'y' : {
                    'mean' : 0.2,
                    'std' : 0.25,
                    'min' : 0,
                    'max' : 0.4,
                    'discrete' : 0.02,
                    # 'reject' : 2.7,
                },
                'theta' : {
                    'mean' : np.pi / 2,
                    'std' : np.pi,
                    'min' : 0,
                    'max' : np.pi,
                    'discrete' : np.pi / 6,
                    # 'reject' : 1.5,
                },
            },
            'rejection_sampling': False,
            'cov_blockdiag': False,
            'smooth_cov': False,
            'nactions': 1,
            'repeat': 1, # number of times to execute the same action
        }
        return hparams

    def _sample_actions_rej(self, M):
        """
        Perform rejection sampling
        :return:
        """
        raise 
        runs = []
        actions = []

        for i in range(M):
            ok = False
            i = 0
            while not ok:
                i +=1
                action_seq = np.random.multivariate_normal(self._mean, self._sigma, 1)

                action_seq = action_seq.reshape(self._hp.nactions, self._adim)
                x_std = self._hp.initial_sampling_configurations['x']['std']
                x_std_fac = self._hp.initial_sampling_configurations['x']['reject']
                y_std = self._hp.initial_sampling_configurations['x']['std']
                y_std_fac = self._hp.initial_sampling_configurations['x']['reject']
                theta_std = self._hp.initial_sampling_configurations['x']['std']
                theta_std_fac = self._hp.initial_sampling_configurations['x']['reject']

                
                if np.any(action_seq[:, 0] - self._mean[0]  > x_std*x_std_fac) or \
                        np.any(action_seq[:, 0] < -xy_std*std_fac) or \
                        np.any(action_seq[:, 1] < -xy_std*std_fac) or \
                        np.any(action_seq[:, 1] < -xy_std*std_fac) or \
                        np.any(action_seq[:, 2] > lift_std*std_fac) or \
                        np.any(action_seq[:, 2] < -lift_std*std_fac):
                    ok = False
                else: ok = True

            runs.append(i)
            actions.append(action_seq)
        actions = np.stack(actions, axis=0)
        if self._hp.discrete_ind != None:
            actions = discretize(actions, self._hp)
        actions = np.repeat(actions, self._hp.repeat, axis=1)
        return actions
        