from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    sfrew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_sft = 0
    cur_ep_len = 0
    ep_rets = []
    ep_sfts = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    sfrews = np.zeros(horizon, 'float32')
    sfpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, sfpred, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_sfts" : ep_sfts, "ep_lens" : ep_lens, 
                    "sfrew" : sfrews, "sfpred" : sfpreds, "nextsfpred": sfpred * (1 - new)}
            _, vpred, sfpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_sfts = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        sfpreds[i] = sfpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, info = env.step(ac)
        sfrew = info[0]["safety_reward"]
        rews[i] = rew
        sfrews[i] = sfrew

        cur_ep_ret += rew
        cur_ep_sft += sfrew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_sfts.append(cur_ep_sft)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_sft = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_targ_and_adv(seg, gamma, lam, lam_sf):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    sfpred = np.append(seg["sfpred"], seg["nextsfpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    seg["sfadv"] = sfgaelam = np.empty(T, 'float32')
    rew, sfrew = seg["rew"], seg["sfrew"]
    lastgaelam, sflastgaelam = 0, 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

        sfdelta = sfrew[t] + gamma * sfpred[t+1] * nonterminal - sfpred[t]
        sfgaelam[t] = sflastgaelam = sfdelta + gamma * lam_sf * nonterminal * sflastgaelam

    seg["tdlamret"] = seg["adv"] + seg["vpred"]
    seg["tdlamsft"] = seg["sfadv"] + seg["sfpred"]
    seg["sfeval"] = np.mean(seg["ep_sfts"])
    seg["T"] = np.mean(seg["ep_lens"])

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    sfatarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function of constraints (if applicable)
    sft = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical safety return

    T = tf.placeholder(dtype=tf.float32, shape=None) # Average timesteps per episode
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    surrsafety = tf.reduce_mean(ratio * sfatarg) * T
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    sf_loss = tf.reduce_mean(tf.square(pi.spred - sft))
    total_loss = pol_surr + pol_entpen + vf_loss + sf_loss
    losses = [pol_surr, pol_entpen, surrsafety, vf_loss, sf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "surrsafety", "vf_loss", "sf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    input_vars = [ob, ac, atarg, sfatarg, ret, sft, T, lrmult]
    lossandgrad = U.function(input_vars, losses + [U.flatgrad(total_loss, var_list)])
    safeandgrad = U.function(input_vars, [surrsafety, U.flatgrad(surrsafety, var_list)])

    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function(input_vars, losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    sftbuffer = deque(maxlen=100) # rolling buffer for episode safety

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_targ_and_adv(seg, gamma, lam, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, sfatarg, T = seg["ob"], seg["ac"], seg["adv"], seg["sfadv"], seg["T"]
        vpredbefore, spredbefore = seg["vpred"], seg["spred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        sfatarg = (sfatarg - sfatarg.mean()) / sfatarg.std() # standardized safety advantage function estimate
        tdlamret, tdlamsft = seg["tdlamret"], seg["tdlamsft"]
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, sfatarg=sfatarg, vtarg=tdlamret, starg=tdlamsft, T=T), deterministic=pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                args = batch["ob"], batch["ac"], batch["atarg"], batch["sfatarg"], batch["vtarg"], batch["starg"], seg["T"], cur_lrmult
                *newlosses, g = lossandgrad(*args)
                _, b = safeandgrad(*args)
                adam.update(g+0.01*b, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            args = batch["ob"], batch["ac"], batch["atarg"], batch["sfatarg"], batch["vtarg"], batch["starg"], seg["T"], cur_lrmult
            newlosses = compute_losses(*args)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
