# CQL

Code for Conservative Q-Learning for Offline Reinforcement Learning (https://arxiv.org/abs/2006.04779)

In this repository we provide code for CQL algorithm described in the paper linked above. We provide code in two sub-directories: `atari` containing code for Atari experiments and `d4rl` containing code for D4RL experiments. Due to changes in the datasets in D4RL, we expect some changes in CQL performance on the new D4RL datasets and we will soon provide a table with new performance numbers for CQL here in this README. We will continually keep updating the numbers here.

If you find this repository useful for your research, please cite:

```
@article{kumar2020conservative,
  author       = {Aviral Kumar and Aurick Zhou and George Tucker and Sergey Levine},
  title        = {Conservative Q-Learning for Offline Reinforcement Learning},
  conference   = {arXiv Pre-print},
  url          = {https://arxiv.org/abs/2006.04779},
}
```

## Atari Experiments
Our code is built on top of the [batch_rl](https://github.com/google-research/batch_rl) repository. Please run installation instructions from the batch_rl repository. CQL in this case was implemented on top of QR-DQN for which the implementation is present in `batch_rl/multi_head/quantile_agent.py`. 

To run experiments in the paper, you will have to specify the size of an individual replay buffer for the purpose of being able to use 1% and 10% data. This is specified in line 53 in `batch_rl/fixed_replay/replay_memory/fixed_replay_memory.py`. For 1%, set `args[2]=1000` and for 10% set `args[2] = 10000`. Depending upon the availability of RAM, you may be able to raise the value of `num_buffers` from 10 to 50 (we were able to do this for 1% datasets) and then change this value in: `self._load_replay_buffers(num_buffers=<>)`.

Now, to run CQL, use the follwing command:

```
python -um batch_rl.fixed_replay.train \
  --base_dir=/tmp/batch_rl \
  --replay_dir=$DATA_DIR/Pong/1 \
  --agent_name=quantile \
  --gin_files='batch_rl/fixed_replay/configs/quantile_pong.gin' \
  --gin_bindings='FixedReplayRunner.num_iterations=1000' \
  --gin_bindings='atari_lib.create_atari_environment.game_name = "Pong"'
  --gin_bindings='FixedReplayQuantileAgent.minq_weight=1.0'
```
For 1% data, use `minq_weight=4.0` and for 10% data, use `minq_weight=1.0`. 

## D4RL Experiments
Our code is built off of [rlkit](https://github.com/vitchyr/rlkit). Please install the conda environment for rlkit while making sure to install `torch>=1.1.0`. Please install [d4rl](https://github.com/rail-berkeley/d4rl). Code for the CQL algorithm is present in `rlkit/torch/sac/cql.py`. After this, for running CQL on the MuJoCo environments, run:

```
python examples/cql_mujoco_new.py --env=<d4rl-mujoco-env-with-version e.g. hopper-medium-v0>
        --policy_lr=1e-4 --seed=10 --lagrange_thresh=-1.0 
        --min_q_weight=(5.0 or 10.0) --gpu=<gpu-id> --min_q_version=3
```

In terms of parameters, we have found `min_q_weight=5.0` or `min_q_weight=10.0` along with `policy_lr=1e-4` or `policy_lr=3e-4` to work reasonably fine for the Gym MuJoCo tasks. These parameters are slightly different from the paper (which will be updated soon) due to differences in the D4RL datasets. For sample performance numbers (final numbers to be updated soon), hopper-medium acheives ~3000 return, and hopper-medium-exprt obtains ~1300 return at the end of 500k gradient steps. To run `CQL(\rho)` [i.e. without the importance sampling], set `min_q_version=2`.

For Ant-Maze tasks, please run:
```
python examples/cql_antmaze_new.py --env=antmaze-medium-play-v0 --policy_lr=1e-4 --seed=10
        --lagrange_thresh=5.0 --min_q_wight=5.0 --gpu=<gpu-id> --min_q_version=3
```

In case of any questions, bugs, suggestions or improvements, please feel free to contact me at aviralk@berkeley.edu or open an issue.
