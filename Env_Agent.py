#!/usr/bin/env python
# coding: utf-8

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories.time_step import TimeStep
import pandas as pd
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import pandas as pd
import numpy as np

class TFAccelerometerEnv(py_environment.PyEnvironment):
    def __init__(self, df):
        self._df = df
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=5, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self._df.shape[1:], dtype=np.float64, minimum=-np.inf, maximum=np.inf, name='observation')
        self._data = df.values
        self._current_step = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_step = 0
        self._episode_ended = False
        if self._current_step >= len(self._data):
            self._episode_ended = True
            return ts.termination(self._data[-1], 0.0)
        return ts.restart(self._data[self._current_step])

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        self._current_step += 1
        if self._current_step >= len(self._data):
            self._episode_ended = True
            return ts.termination(self._data[-1], reward=0.0)

        return ts.transition(self._data[self._current_step], reward=0.0, discount=1.0)