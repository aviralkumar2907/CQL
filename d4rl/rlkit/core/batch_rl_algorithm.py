import abc
import copy
# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rlkit.torch import pytorch_util as ptu

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.core.rl_algorithm import eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
import numpy as np
from rlkit.torch.core import np_to_pytorch_batch

import torch

def get_flat_params(model):
    params = []
    for param in model.parameters():
        # import ipdb; ipdb.set_trace()
        params.append(param.data.cpu().numpy().reshape(-1))
    return np.concatenate(params)

def set_flat_params(model, flat_params, trainable_only=True):
    idx = 0
    # import ipdb; ipdb.set_trace()
    for p in model.parameters():
        flat_shape = int(np.prod(list(p.data.shape)))
        flat_params_to_assign = flat_params[idx:idx+flat_shape]
      
        if len(p.data.shape):
            p.data = ptu.tensor(flat_params_to_assign.reshape(*p.data.shape))
        else:
            p.data = ptu.tensor(flat_params_to_assign[0])
        idx += flat_shape  
    return model

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            q_learning_alg=False,
            eval_both=False,
            batch_rl=False,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.batch_rl = batch_rl
        self.q_learning_alg = q_learning_alg
        self.eval_both = eval_both

        ### Reserve path collector for evaluation, visualization
        self._reserve_path_collector = MdpPathCollector(
            env=evaluation_env, policy=self.trainer.policy, 
        )
    
    def policy_fn(self, obs):
        """
        Used when sampling actions from the policy and doing max Q-learning
        """
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            state = ptu.from_numpy(obs.reshape(1, -1)).repeat(10, 1)
            action, _, _, _, _, _, _, _  = self.trainer.policy(state)
            q1 = self.trainer.qf1(state, action)
            ind = q1.max(0)[1]
        return ptu.get_numpy(action[ind]).flatten()
    
    def policy_fn_discrete(self, obs):
        with torch.no_grad():
            obs = ptu.from_numpy(obs.reshape(1, -1))
            q_vector = self.trainer.qf1.q_vector(obs)
            action = q_vector.max(1)[1]
        ones = np.eye(q_vector.shape[1])
        return ptu.get_numpy(action).flatten()


    def _train(self):
        if self.min_num_steps_before_training > 0 and not self.batch_rl:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if self.q_learning_alg:
                policy_fn = self.policy_fn
                if self.trainer.discrete:
                    policy_fn = self.policy_fn_discrete
                self.eval_data_collector.collect_new_paths(
                    policy_fn,
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True
                )
            else:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                if not self.batch_rl:
                    # Sample new paths only if not doing batch rl
                    new_expl_paths = self.expl_data_collector.collect_new_paths(
                        self.max_path_length,
                        self.num_expl_steps_per_train_loop,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)

                    self.replay_buffer.add_paths(new_expl_paths)
                    gt.stamp('data storing', unique=False)
                elif self.eval_both:
                    # Now evaluate the policy here:
                    policy_fn = self.policy_fn
                    if self.trainer.discrete:
                        policy_fn = self.policy_fn_discrete
                    new_expl_paths = self.expl_data_collector.collect_new_paths(
                        policy_fn,
                        self.max_path_length,
                        self.num_eval_steps_per_epoch,
                        discard_incomplete_paths=True,
                    )

                    gt.stamp('policy fn evaluation')

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

            # import ipdb; ipdb.set_trace()
            ## After epoch visualize
            # if epoch % 50 == 0:
            #     self._visualize(policy=True, num_dir=300, alpha=0.05, iter=epoch)
            #     print ('Saved Plots ..... %d'.format(epoch))
    
    def _eval_q_custom_policy(self, custom_model, q_function):
        data_batch = self.replay_buffer.random_batch(self.batch_size)
        data_batch = np_to_pytorch_batch(data_batch)
        return self.trainer.eval_q_custom(custom_model, data_batch, q_function=q_function)
    
    def eval_policy_custom(self, policy):
        """Update policy and then look at how the returns under this policy look like."""
        self._reserve_path_collector.update_policy(policy)
        
        # Sampling
        eval_paths = self._reserve_path_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        # gt.stamp('evaluation during viz sampling')

        eval_returns = eval_util.get_average_returns(eval_paths)
        return eval_returns
    
    def plot_visualized_data(self, array_plus, array_minus, base_val, fig_label='None'):
        """Plot two kinds of visualizations here: 
           (1) Trend of loss_minus with respect to loss_plus
           (2) Histogram of different gradient directions
        """
        # Type (1)
        array_plus = array_plus - base_val
        array_minus = array_minus - base_val
        print (fig_label)
        fig, ax = plt.subplots()
        ax.scatter(array_minus, array_plus)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        # import ipdb; ipdb.set_trace()
        # ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.ylabel('L (theta + alpha * d) - L(theta)')
        plt.xlabel('L (theta - alpha * d) - L(theta)')
        plt.title('Loss vs Loss %s' % fig_label)
        plt.savefig('plots_hopper_correct_online_3e-4_n10_viz_sac_again/type1_' + (fig_label) + '.png')

        # Type (2)
        plt.figure(figsize=(5, 4))
        plt.subplot(211)
        grad_projections = (array_plus - array_minus)*0.5
        plt.hist(grad_projections, bins=50)
        plt.xlabel('Gradient Value')
        plt.ylabel('Count')
        plt.subplot(212)
        
        # Curvature
        curvature_projections = (array_plus + array_minus)*0.5
        plt.hist(curvature_projections, bins=50)
        plt.xlabel('Curvature Value')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('plots_hopper_correct_online_3e-4_n10_viz_sac_again/spectra_joined_' + (fig_label) + '.png')

    def _visualize(self, policy=False, q_function=False, num_dir=50, alpha=0.1, iter=None):
        assert policy or q_function, "Both are false, need something to visualize"
        # import ipdb; ipdb.set_trace()
        policy_weights = get_flat_params(self.trainer.policy)
        # qf1_weights = get_flat_params(self.trainer.qf1)
        # qf2_weights = get_flat_params(self.trainer.qf2)
        
        policy_dim = policy_weights.shape[0]
        # qf_dim = qf1_weights.shape[0]
        
        # Create clones to assign weights
        policy_clone = copy.deepcopy(self.trainer.policy)

        # Create arrays for storing data
        q1_plus_eval = []
        q1_minus_eval = []
        q2_plus_eval = []
        q2_minus_eval = []
        qmin_plus_eval = []
        qmin_minus_eval = []
        returns_plus_eval = []
        returns_minus_eval = []

        # Groundtruth policy params
        policy_eval_qf1 = self._eval_q_custom_policy(self.trainer.policy, self.trainer.qf1)
        policy_eval_qf2 = self._eval_q_custom_policy(self.trainer.policy, self.trainer.qf2)
        policy_eval_q_min = min(policy_eval_qf1, policy_eval_qf2)
        policy_eval_returns = self.eval_policy_custom(self.trainer.policy)

        # These are the policy saddle point detection
        for idx in range(num_dir):
            random_dir = np.random.normal(size=(policy_dim))
            theta_plus = policy_weights + alpha * policy_dim
            theta_minus = policy_weights - alpha * policy_dim

            set_flat_params(policy_clone, theta_plus)
            q_plus_1 = self._eval_q_custom_policy(policy_clone, self.trainer.qf1)
            q_plus_2 = self._eval_q_custom_policy(policy_clone, self.trainer.qf2)
            q_plus_min = min(q_plus_1, q_plus_2)
            eval_return_plus = self.eval_policy_custom(policy_clone)

            set_flat_params(policy_clone, theta_minus)
            q_minus_1 = self._eval_q_custom_policy(policy_clone, self.trainer.qf1)
            q_minus_2 = self._eval_q_custom_policy(policy_clone, self.trainer.qf2)
            q_minus_min = min(q_minus_1, q_minus_2)
            eval_return_minus = self.eval_policy_custom(policy_clone)

            # Append to array
            q1_plus_eval.append(q_plus_1)
            q2_plus_eval.append(q_plus_2)
            q1_minus_eval.append(q_minus_1)
            q2_minus_eval.append(q_minus_2)
            qmin_plus_eval.append(q_plus_min)
            qmin_minus_eval.append(q_minus_min)
            returns_plus_eval.append(eval_return_plus)
            returns_minus_eval.append(eval_return_minus)
        
        # Now we visualize
        # import ipdb; ipdb.set_trace()

        q1_plus_eval = np.array(q1_plus_eval)
        q1_minus_eval = np.array(q1_minus_eval)
        q2_plus_eval = np.array(q2_plus_eval)
        q2_minus_eval = np.array(q2_minus_eval)
        qmin_plus_eval = np.array(qmin_plus_eval)
        qmin_minus_eval = np.array(qmin_minus_eval)
        returns_plus_eval = np.array(returns_plus_eval)
        returns_minus_eval = np.array(returns_minus_eval)

        self.plot_visualized_data(q1_plus_eval, q1_minus_eval, policy_eval_qf1, fig_label='q1_policy_params_iter_' + (str(iter)))
        self.plot_visualized_data(q2_plus_eval, q2_minus_eval, policy_eval_qf2, fig_label='q2_policy_params_iter_' + (str(iter)))
        self.plot_visualized_data(qmin_plus_eval, qmin_minus_eval, policy_eval_q_min, fig_label='qmin_policy_params_iter_' + (str(iter)))
        self.plot_visualized_data(returns_plus_eval, returns_minus_eval, policy_eval_returns, fig_label='returns_policy_params_iter_' + (str(iter)))

        del policy_clone
