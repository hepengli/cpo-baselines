import time
from collections import deque
import tensorflow as tf, numpy as np

from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.input import observation_placeholder
from baselines.pcpo.policies import build_policy
from baselines.pcpo.runner import Runner
from contextlib import contextmanager

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def learn(*,
        network,
        env,
        nsteps,
        total_timesteps,
        max_kl=0.01,
        max_sf=1e10,
        gamma=0.995,
        lam=0.95, # advantage estimation
        ent_coef=0.0,
        cg_iters=10,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters=3,
        max_episodes=0,
        max_iters=0,
        seed=None,
        model_fn=None,
        callback=None,
        load_path=None,
        is_finite=True,
        **network_kwargs
        ):
    '''
    learn a policy function with Parallel CPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    total_timesteps         max number of timesteps

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    max_sf                  max safety constraint value

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''
    # Setup stuff
    # ----------------------------------------
    set_global_seeds(seed)
    np.set_printoptions(precision=3)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    ob_space = env.observation_space
    ac_space = env.action_space

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.pcpo.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, ent_coef=ent_coef,
                vf_stepsize=vf_stepsize, vf_iters=vf_iters, load_path=load_path, 
                cg_damping=cg_damping, cg_iters=cg_iters, max_kl=max_kl, max_sf=max_sf)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, is_finite=is_finite)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    sftbuffer = deque(maxlen=40) # rolling buffer for episode safety

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return model

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with model.timed("sampling"):
            obs, returns, masks, actions, values, advs, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
            ep_lens, ep_rets, ep_sfts = [],[],[]
            for info in epinfos:
                ep_lens.append(info['l'])
                ep_rets.append(info['r'])
                ep_sfts.append(info['s'])

        model.train(obs, returns, masks, actions, values, advs, neglogpacs, states, ep_lens, ep_rets, ep_sfts)

        lrlocal = (ep_lens, ep_rets, ep_sfts) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews, sfts = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.append(np.mean(lens))
        rewbuffer.append(np.mean(rews))
        sftbuffer.append(np.mean(sfts))

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpSftMean", np.mean(sftbuffer))

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.dump_tabular()

    return model.pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]