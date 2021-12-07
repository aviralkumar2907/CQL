# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logged Replay Buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from concurrent import futures
from dopamine.replay_memory import circular_replay_buffer
from batch_rl.baselines.replay_memory import logged_replay_buffer

import numpy as np
import tensorflow.compat.v1 as tf

import gin
gfile = tf.gfile

STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX


class FixedReplayBuffer(object):
  """Object composed of a list of OutofGraphReplayBuffers."""

  def __init__(self, data_dir, replay_suffix, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Initialize the FixedReplayBuffer class.

    Args:
      data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      *args: Arbitrary extra arguments.
      **kwargs: Arbitrary keyword arguments.
    """
    self._args = args

    # For 1% experiments, we load 10000 sized buffers
    self._args = list(self._args)
    self._args[2] = 10000
    self._args = tuple(self._args)

    self._kwargs = kwargs
    self._data_dir = data_dir
    self._loaded_buffers = False
    self.add_count = np.array(0)
    self._replay_suffix = replay_suffix
    while not self._loaded_buffers:
      if replay_suffix:
        assert replay_suffix >= 0, 'Please pass a non-negative replay suffix'
        self.load_single_buffer(replay_suffix)
      else:
        # self._load_and_save_buffers(num_load=2, num_buffers=50)

        # For 1% experiments, set num_buffers=50
        self._load_replay_buffers(num_buffers=10)

        # For first 20M samples
        # self._load_replay_buffers_initial(num_load=5, num_buffers=5)

  def load_single_buffer(self, suffix):
    """Load a single replay buffer."""
    print ('Load single buffer')
    replay_buffer = self._load_buffer(suffix)
    if replay_buffer is not None:
      self._replay_buffers = [replay_buffer]
      self.add_count = replay_buffer.add_count
      self._num_replay_buffers = 1
      self._loaded_buffers = True

  def _load_buffer(self, suffix):
    """Loads a OutOfGraphReplayBuffer replay buffer."""
    try:
      # pytype: disable=attribute-error
      replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
          *self._args, **self._kwargs)
      print ('Loading buffer in fixed replay buffer')
      replay_buffer.load(self._data_dir, suffix)
      tf.logging.info('Loaded replay buffer ckpt {} from {}'.format(
          suffix, self._data_dir))
      # pytype: enable=attribute-error
      return replay_buffer
    except tf.errors.NotFoundError:
      print ('Exception')
      return None
  
  def _load_replay_buffers_initial(self, num_load=None, num_buffers=None):
    """Load multiple checkpoints into a list of replay buffers."""
    if not self._loaded_buffers:
      ckpts = gfile.ListDirectory(self._data_dir)
      ckpt_counters = collections.Counter(
        [name.split('.')[-2] for name in ckpts])
      
      ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
      ckpt_indices = list([int(x) for x in ckpt_suffixes])
      list_indices = np.arange(len(ckpt_indices))
      ckpt_index_dict = dict(zip(ckpt_indices, list_indices))
      # import ipdb; ipdb.set_trace()
      if num_buffers is not None:
        if num_load is None:
          num_load = 1
        ckpt_index = np.random.choice(
          np.arange(num_buffers), num_load, replace=False
        )
        list_index = [ckpt_index_dict[x] for x in ckpt_index]
        ckpt_suffixes = [ckpt_suffixes[idx] for idx in list_index]
      self._replay_buffers = []

      with futures.ThreadPoolExecutor(
          max_workers=num_buffers) as thread_pool_executor:
        print ('Ckpt suffixes initial: ', ckpt_suffixes)
        replay_futures = [thread_pool_executor.submit(
            self._load_buffer, suffix) for suffix in ckpt_suffixes]
      for f in replay_futures:
        replay_buffer = f.result()
        if replay_buffer is not None:
          self._replay_buffers.append(replay_buffer)
          self.add_count = max(replay_buffer.add_count, self.add_count)
      self._num_replay_buffers = len(self._replay_buffers)
      if self._num_replay_buffers:
        self._loaded_buffers = True
  
  def _load_and_save_buffers(self, num_load=None, num_buffers=None):
    # import ipdb; ipdb.set_trace()
    ckpts = gfile.ListDirectory(self._data_dir)
    ckpt_counters = collections.Counter(
      [name.split('.')[-2] for name in ckpts])
    
    ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
    num_load_steps = int(num_buffers // num_load)
    for idx in range(num_load_steps):
      ckpt_indices = ckpt_suffixes[idx * num_load: (idx + 1) * num_load]
      with futures.ThreadPoolExecutor(
            max_workers=num_load) as thread_pool_executor:
        print ('Ckpt suffixes: ', ckpt_indices)
        replay_futures = [thread_pool_executor.submit(
            self._load_and_save_buffer, suffix) for suffix in ckpt_indices]
        for f in replay_futures:
          replay_buffer = f.result()
      print ('Done: ', idx, num_load_steps)

  def _load_replay_buffers(self, num_buffers=None):
    """Loads multiple checkpoints into a list of replay buffers."""
    if not self._loaded_buffers:  # pytype: disable=attribute-error
      ckpts = gfile.ListDirectory(self._data_dir)  # pytype: disable=attribute-error
      # Assumes that the checkpoints are saved in a format CKPT_NAME.{SUFFIX}.gz
      ckpt_counters = collections.Counter(
          [name.split('.')[-2] for name in ckpts])
      # Should contain the files for add_count, action, observation, reward,
      # terminal and invalid_range
      ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
      if num_buffers is not None and len(ckpt_suffixes) > 0:
        ckpt_suffixes = np.random.choice(
            ckpt_suffixes, num_buffers, replace=False)
      self._replay_buffers = []
      # Load the replay buffers in parallel
      with futures.ThreadPoolExecutor(
          max_workers=num_buffers) as thread_pool_executor:
        print ('Ckpt suffixes: ', ckpt_suffixes)
        replay_futures = [thread_pool_executor.submit(
            self._load_buffer, suffix) for suffix in ckpt_suffixes]
      for f in replay_futures:
        replay_buffer = f.result()
        if replay_buffer is not None:
          self._replay_buffers.append(replay_buffer)
          self.add_count = max(replay_buffer.add_count, self.add_count)
      self._num_replay_buffers = len(self._replay_buffers)
      print ('Number of replay buffers: ', self._num_replay_buffers)
      if self._num_replay_buffers:
        self._loaded_buffers = True

  def get_transition_elements(self):
    return self._replay_buffers[0].get_transition_elements()

  def sample_transition_batch(self, batch_size=None, indices=None):
    buffer_index = np.random.randint(self._num_replay_buffers)
    return self._replay_buffers[buffer_index].sample_transition_batch(
        batch_size=batch_size, indices=indices)

  def load(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def reload_buffer(self, num_buffers=None):
    pass
    # self._loaded_buffers = False
    # self._load_replay_buffers(num_buffers=10)
    # self._load_replay_buffers_initial(num_load=2, num_buffers=5)

  def save(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def add(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedFixedReplayBuffer(circular_replay_buffer.WrappedReplayBuffer):
  """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism."""

  def __init__(self,
               data_dir,
               replay_suffix,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    """Initializes WrappedFixedReplayBuffer."""

    memory = FixedReplayBuffer(
        data_dir, replay_suffix, observation_shape, stack_size, replay_capacity,
        batch_size, update_horizon, gamma, max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype)

    super(WrappedFixedReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging=use_staging,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        wrapped_memory=memory,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)
