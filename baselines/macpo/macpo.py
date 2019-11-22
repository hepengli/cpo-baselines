import time
from collections import deque
import tensorflow as tf, numpy as np

from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.input import observation_placeholder
from baselines.macpo.policies import build_policy
from baselines.macpo.model import Model
from baselines.macpo.runner import Runner
from contextlib import contextmanager

import gym
from baselines.bench import Monitor
from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers, set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        try:
            low, high = self.env.unwrapped._feasible_action()
        except:
            low, high = self.action_space.low, self.action_space.high

        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, low, high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class PCPO(object):
    """ Paralell CPO algorithm """
    def __init__(self, network, env, nsteps, max_kl=0.01, max_sf=1e10, gamma=0.995, lam=0.95, ent_coef=0.0, 
                cg_iters=10, cg_damping=1e-2, qf_stepsize=3e-4, qf_iters=3, vf_stepsize=3e-4, vf_iters=3, 
                num_env=1, seed=None, load_path=None, logger_dir=None, is_finite=True, name_scope=None, **network_kwargs):
        # Setup stuff
        set_global_seeds(seed)
        np.set_printoptions(precision=3)

        if isinstance(env, str):
            env = self.make_vec_env(env, seed=seed, logger_dir=logger_dir, reward_scale=1.0, num_env=num_env)

        ob_space = env.observation_space
        ac_space = env.action_space

        policy = build_policy(env, network, **network_kwargs)

        # Instantiate the model object and runner object
        model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, ent_coef=ent_coef, name_scope=name_scope,
                      qf_stepsize=qf_stepsize, qf_iters=qf_iters, vf_stepsize=vf_stepsize, vf_iters=vf_iters, 
                      load_path=load_path, cg_damping=cg_damping, cg_iters=cg_iters, max_kl=max_kl, max_sf=max_sf)
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, is_finite=is_finite)

        self.env = env
        self.model = model
        self.runner = runner
        self.total_episodes = 0
        self.total_timesteps = 0
        self.total_iters = 0
        self.tstart = time.time()
        self.lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
        self.sftbuffer = deque(maxlen=40) # rolling buffer for episode safety

    def train(self, timesteps=0, episodes=0, iters=0):
        if sum([timesteps>0, episodes>0, iters>0])==0:
            # noththing to be done
            return

        assert sum([timesteps>0, episodes>0, iters>0]) < 2, \
            'out of iters, timesteps, and episodes only one should be specified'

        timesteps_so_far, episodes_so_far, iters_so_far = 0, 0, 0
        while True:
            if timesteps and timesteps_so_far >= timesteps:
                break
            elif episodes and episodes_so_far >= episodes:
                break
            elif iters and iters_so_far >= iters:
                break
            logger.log("********** Iteration %i ************"%self.total_iters)

            with self.model.timed("sampling"):
                obs, returns, masks, actions, values, advs, neglogpacs, oobs, q_values, states, epinfos,  = self.runner.run() #pylint: disable=E0632
                ep_lens, ep_rets, ep_sfts = [],[],[]
                for info in epinfos:
                    ep_lens.append(info['l'])
                    ep_rets.append(info['r'])
                    ep_sfts.append(info['s'])

            self.model.train(obs, returns, masks, actions, values, advs, neglogpacs, oobs, q_values, states, ep_lens, ep_rets, ep_sfts)

            lrlocal = (ep_lens, ep_rets, ep_sfts) # local values
            if MPI is not None:
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            else:
                listoflrpairs = [lrlocal]

            lens, rews, sfts = map(flatten_lists, zip(*listoflrpairs))
            self.lenbuffer.append(np.mean(lens))
            self.rewbuffer.append(np.mean(rews))
            self.sftbuffer.append(np.mean(sfts))

            logger.record_tabular("EpLenMean", np.mean(self.lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(self.rewbuffer))
            logger.record_tabular("EpSftMean", np.mean(self.sftbuffer))

            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1

            self.total_episodes += len(lens)
            self.total_timesteps += sum(lens)
            self.total_iters += 1

            logger.record_tabular("EpisodesSoFar", self.total_episodes)
            logger.record_tabular("TimestepsSoFar", self.total_timesteps)
            logger.record_tabular("TimeElapsed", time.time() - self.tstart)
            logger.dump_tabular()

            # self.env.envs[0].unwrapped.render()

    def make_env(self, env_id, seed, train=True, logger_dir=None, reward_scale=1.0, mpi_rank=0, subrank=0):
        """
        Create a wrapped, monitored gym.Env for safety.
        """
        env = gym.make(env_id, **{"train":train})
        env.seed(seed + subrank if seed is not None else None)
        env = Monitor(env, 
                    logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                    allow_early_resets=True)
        env.seed(seed)
        env = ClipActionsWrapper(env)
        if reward_scale != 1.0:
            from baselines.common.retro_wrappers import RewardScaler
            env = RewardScaler(env, reward_scale)
        return env

    def make_vec_env(self, env_id, seed, train=True, logger_dir=None, reward_scale=1.0, num_env=1):
        """
        Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
        """
        mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        seed = seed + 10000 * mpi_rank if seed is not None else None
        def make_thunk(rank, initializer=None):
            return lambda: self.make_env(
                env_id,
                seed,
                train=True,
                logger_dir=logger_dir,
                reward_scale=reward_scale,
                mpi_rank=mpi_rank,
                subrank=0
            )
        set_global_seeds(seed)

        return DummyVecEnv([make_thunk(i) for i in range(num_env)])

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]