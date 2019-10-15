from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
import time
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.cpo.policies import build_policy
from contextlib import contextmanager

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    if isinstance(ac, tuple):
        ac = np.concatenate(list(ac))
    new = True
    rew = [0.0, 0.0]
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_sft = 0
    cur_ep_len = 0
    ep_rets = []
    ep_sfts = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    rews = np.zeros([horizon, 2], 'float32')
    vpreds = np.zeros([horizon, 2], 'float32')
    news = np.zeros(horizon, 'int32')
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_sfts" : ep_sfts, "ep_lens" : ep_lens, 
                   }
            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_sfts = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        vpreds[i] = vpred
        prevacs[i] = prevac

        ob, rew, new, info = env.step(ac)
        if isinstance(env, VecEnv):
            sft = info[0]["s"] if "s" in info[0].keys() else 0.0
        else:
            sft = info["s"] if "s" in info.keys() else 0.0
        rews[i] = [rew, sft]

        cur_ep_ret += rew
        cur_ep_sft += sft
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

def add_targ_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.vstack([seg["vpred"], seg["nextvpred"]])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty([T, 2], 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["tdlamret"] = seg["adv"] + seg["vpred"]
    seg["sfeval"] = np.mean(seg["ep_sfts"])
    seg["T"] = np.mean(seg["ep_lens"])

def learn(*,
        network,
        env,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        max_kl=0.01,
        max_sf=1e10,
        cg_iters=10,
        gamma=0.995,
        lam=0.95, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters=3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        comp_threshold=True,
        diff_threshold=False,
        attempt_feasible_recovery=True,
        attempt_infeasible_recovery=True,
        revert_to_last_safe_point=False,
        linesearch_infeasible_recovery=True,
        accept_violation=False,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

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

    cg_iters                number of iterations of conjugate gradient algorithm

    ent_coef                coefficient of policy entropy term in the optimization objective

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

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    policy = build_policy(env, network, **network_kwargs)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None, 2]) # Target advantage function of objective (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None, 2]) # Empirical return of rewards and safeties
    T = tf.placeholder(dtype=tf.float32, shape=None) # Average timesteps per episode

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrloss = - tf.reduce_mean(ratio * atarg[:,0])
    surrsafe = tf.reduce_mean(ratio * atarg[:,1]) * T
    losses = surrloss - entbonus
    losssafety = [losses, meankl, surrsafe, entbonus, surrloss, meanent]
    losssafety_names = ["losses", "meankl", "surrsafe", "entbonus", "surrloss", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

    compute_losssafety = U.function([ob, ac, atarg, T], losssafety)
    compute_losssandgrad = U.function([ob, ac, atarg, T], [losses, U.flatgrad(losses, var_list)])
    compute_safetyandgrad = U.function([ob, ac, atarg, T], [surrsafe, U.flatgrad(surrsafe, var_list)])
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
    compute_fvp = U.function([flat_tangent, ob, ac, atarg, T], fvp)


    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out

    U.initialize()
    if load_path is not None:
        pi.load(load_path)

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    global episodes_so_far, timesteps_so_far, iters_so_far
    eps = 1e-8
    backtrack_ratio = 0.8
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    sftbuffer = deque(maxlen=40) # rolling buffer for episode safety

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

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

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_targ_and_adv(seg, gamma, lam)

        # ob, ac, atarg, sfatarg, T = map(np.concatenate, (obs, acs, atargs, sfatarg, T))
        ob, ac, atarg, T = seg["ob"], seg["ac"], seg["adv"], seg["T"]
        atarg = (atarg - atarg.mean()) / (atarg.std() + eps) # standardized advantage function estimate

        args = seg["ob"], seg["ac"], atarg, seg["T"]
        fvpargs = [arr[::5] for arr in args[:3]]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        # set old parameter values to new parameter values
        assign_old_eq_new()

        # update value network
        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        # compute loss and safety gradients
        with timed("computegrad"):
            lossbefore, g = compute_losssandgrad(*args)
            safetybefore, b = compute_safetyandgrad(*args)
        g, b = allmean(g), allmean(b)

        # compute search direction
        with timed("cg"):
            v = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
        approx_g = fisher_vector_product(v)
        q = v.dot(approx_g) # approx = g^T H^{-1} g
        delta = 2 * max_kl

        # residual = np.sqrt((approx_g - g).dot(approx_g - g))
        # rescale  = q / (v.dot(v))
        # logger.record_tabular("OptimDiagnostic_Residual",residual)
        # logger.record_tabular("OptimDiagnostic_Rescale", rescale)

        S = seg["sfeval"]
        c = S - max_sf

        if c > 0:
            logger.log("warning! safety constraint is already violated")
        else:
            # the current parameters constitute a feasible point: save it as "last good point"
            th_lastsafe = get_flat()

        # can't stop won't stop (unless something in the conditional checks / calculations that follow
        # require premature stopping of optimization process)
        stop_flag = False

        if b.dot(b) <= eps:
            # if safety gradient is zero, linear constraint is not present;
            # ignore its implementation.
            lm = np.sqrt(q / delta)
            nu = 0
            w = 0
            r,s,A,B = 0,0,0,0
            optim_case = 4
        else:
            norm_b = np.sqrt(b.dot(b))
            unit_b = b / norm_b
            w = norm_b * cg(fisher_vector_product, unit_b, cg_iters=cg_iters, verbose=rank==0)

            r = w.dot(approx_g)                    # approx = b^T H^{-1} g
            s = w.dot(fisher_vector_product(w))    # approx = b^T H^{-1} b

            # figure out lambda coeff (lagrange multiplier for trust region)
            # and nu coeff (lagrange multiplier for linear constraint)
            A = q - r**2 / s                # this should always be positive by Cauchy-Schwarz
            B = delta - c**2 / s            # this one says whether or not the closest point on the plane is feasible

            # if (B < 0), that means the trust region plane doesn't intersect the safety boundary
            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B > 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c > 0 and B > 0:
                # x = 0 is infeasible (bad! unsafe!) and safety boundary intersects
                # ==> part of trust region is feasible
                # ==> this is 'recovery mode'
                optim_case = 1
                if attempt_feasible_recovery:
                    logger.log("alert! conjugate constraint optimizer is attempting feasible recovery")
                else:
                    logger.log("alert! problem is feasible but needs recovery, and we were instructed not to attempt recovery")
                    stop_flag = True
            else:
                # x = 0 infeasible (bad! unsafe!) and safety boundary doesn't intersect
                # ==> whole trust region infeasible
                # ==> optimization problem infeasible!!!
                optim_case = 0
                if attempt_infeasible_recovery:
                    logger.log("alert! conjugate constraint optimizer is attempting infeasible recovery")
                else:
                    logger.log("alert! problem is infeasible, and we were instructed not to attempt recovery")
                    stop_flag = True


            # default dual vars, which assume safety constraint inactive
            # (this corresponds to either optim_case == 3,
            #  or optim_case == 2 under certain conditions)
            lm = np.sqrt(q / delta)
            nu  = 0

            if optim_case == 2 or optim_case == 1:

                # dual function is piecewise continuous
                # on region (a):
                #
                #   L(lm) = -1/2 (A / lm + B * lm) - r * c / s
                # 
                # on region (b):
                #
                #   L(lm) = -1/2 (q / lm + delta * lm)
                # 

                lm_mid = r / c
                L_mid = - 0.5 * (q / lm_mid + lm_mid * delta)

                lm_a = np.sqrt(A / (B + eps))
                L_a = -np.sqrt(A*B) - r*c / (s + eps)                 
                # note that for optim_case == 1 or 2, B > 0, so this calculation should never be an issue

                lm_b = np.sqrt(q / delta)
                L_b = -np.sqrt(q * delta)

                #those lm's are solns to the pieces of piecewise continuous dual function.
                #the domains of the pieces depend on whether or not c < 0 (x=0 feasible),
                #and so projection back on to those domains is determined appropriately.
                if lm_mid > 0:
                    if c < 0:
                        # here, domain of (a) is [0, lm_mid)
                        # and domain of (b) is (lm_mid, infty)
                        if lm_a > lm_mid:
                            lm_a = lm_mid
                            L_a   = L_mid
                        if lm_b < lm_mid:
                            lm_b = lm_mid
                            L_b   = L_mid
                    else:
                        # here, domain of (a) is (lm_mid, infty)
                        # and domain of (b) is [0, lm_mid)
                        if lm_a < lm_mid:
                            lm_a = lm_mid
                            L_a   = L_mid
                        if lm_b > lm_mid:
                            lm_b = lm_mid
                            L_b   = L_mid

                    if L_a >= L_b:
                        lm = lm_a
                    else:
                        lm = lm_b

                else:
                    if c < 0:
                        lm = lm_b
                    else:
                        lm = lm_a

                nu = max(0, lm * c - r) / (s + eps)

        logger.record_tabular("OptimCase", optim_case)  # 4 / 3: trust region totally in safe region; 
                                                        # 2 : trust region partly intersects safe region, and current point is feasible
                                                        # 1 : trust region partly intersects safe region, and current point is infeasible
                                                        # 0 : trust region does not intersect safe region
        logger.record_tabular("LagrangeLamda", lam) # dual variable for trust region
        logger.record_tabular("LagrangeNu", nu)     # dual variable for safety constraint
        # logger.record_tabular("OptimDiagnostic_q",q) # approx = g^T H^{-1} g
        # logger.record_tabular("OptimDiagnostic_r",r) # approx = b^T H^{-1} g
        # logger.record_tabular("OptimDiagnostic_s",s) # approx = b^T H^{-1} b
        # logger.record_tabular("OptimDiagnostic_c",c) # if > 0, constraint is violated
        # logger.record_tabular("OptimDiagnostic_A",A) 
        # logger.record_tabular("OptimDiagnostic_B",B)
        logger.record_tabular("OptimDiagnostic_S",S)
        if nu == 0:
            logger.log("safety constraint is not active!")

        # Predict worst-case next S
        nextS = S + np.sqrt(delta * s)
        logger.record_tabular("OptimDiagnostic_WorstNextS",nextS)


        if optim_case > 0:
            fullstep = (1. / (lm + eps) ) * ( v + nu * w )
        else:
            # current default behavior for attempting infeasible recovery:
            # take a step on natural safety gradient
            fullstep = np.sqrt(delta / (s + eps)) * w

        logger.log("descent direction computed")

        thbefore = get_flat()

        # Safety thershold
        threshold = max_sf
        if comp_threshold:
            threshold = max(max_sf - seg["sfeval"], 0)
            threshold = max(threshold - 1.0 * np.std(seg["adv"][:,1]), 0)
        if diff_threshold:
            threshold += safetybefore
        logger.record_tabular("OptimDiagnostic_Threshold", threshold)

        def check_nan():
            loss, kl, safety, *_ = compute_losssafety(*args)
            if np.isnan(loss) or np.isnan(kl) or np.isnan(safety):
                logger.log("Something is NaN. Rejecting the step!")
                if np.isnan(loss):
                    logger.log("Violated because loss is NaN")
                if np.isnan(kl):
                    logger.log("Violated because kl is NaN")
                if np.isnan(safety):
                    logger.log("Violated because safety is NaN")
                set_from_flat(thbefore)


        def line_search(check_loss=True, check_kl=True, check_safety=True):
            loss_rejects = 0
            kl_rejects = 0
            safety_rejects  = 0
            n_iter = 0
            for n_iter, ratio in enumerate(backtrack_ratio ** np.arange(15)):
                thnew = thbefore - ratio * fullstep
                set_from_flat(thnew)
                loss, kl, safety, *_ = compute_losssafety(*args)
                loss_flag = loss < lossbefore
                kl_flag = kl <= max_kl
                safety_flag  = safety <= threshold
                if check_loss and not(loss_flag):
                    logger.log("At backtrack itr %i, loss failed to improve." % n_iter)
                    loss_rejects += 1
                if check_kl and not(kl_flag):
                    logger.log("At backtrack itr %i, KL-Divergence violated." % n_iter)
                    logger.log("KL-Divergence violation was %.3f %%." % (100*(kl / max_kl) - 100))
                    kl_rejects += 1
                if check_safety and not(safety_flag):
                    logger.log("At backtrack itr %i, expression for safety constraint failed to improve." % n_iter)
                    logger.log("Safety constraint violation was %.3f %%." % (100*(safety / threshold) - 100))
                    safety_rejects += 1
                if (loss_flag or not(check_loss)) and (kl_flag or not(check_kl)) and (safety_flag or not(check_safety)):
                    logger.log("Accepted step at backtrack itr %i." % n_iter)
                    break

            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

            return loss, kl, safety, n_iter


        def wrap_up():
            global episodes_so_far, timesteps_so_far, iters_so_far
            lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_sfts"]) # local values
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

            loss, kl, safety, _, _, entropy = compute_losssafety(*args)
            logger.record_tabular("EvalLoss", loss)
            logger.record_tabular("EvalSafety", safety)
            logger.record_tabular("EvalKL", kl)
            logger.record_tabular("EvalEntropy", entropy)

            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1

            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            logger.dump_tabular()

        if stop_flag==True:
            wrap_up()
            continue

        if optim_case == 1 and not(revert_to_last_safe_point):
            if linesearch_infeasible_recovery:
                logger.log("feasible recovery mode: constrained natural gradient step. performing linesearch on constraints.")
                loss, kl, safety, n_iter = line_search(False,True,True)
            else:
                set_from_flat(thbefore-fullstep)
                logger.log("feasible recovery mode: constrained natural gradient step. no linesearch performed.")
            check_nan()
            wrap_up()
            continue
        elif optim_case == 0 and not(revert_to_last_safe_point):
            if linesearch_infeasible_recovery:
                logger.log("infeasible recovery mode: natural safety step. performing linesearch on constraints.")
                loss, kl, safety, n_iter = line_search(False,True,True)
            else:
                set_from_flat(thbefore-fullstep)
                logger.log("infeasible recovery mode: natural safety gradient step. no linesearch performed.")
            check_nan()
            wrap_up()
            continue
        elif (optim_case == 0 or optim_case == 1) and revert_to_last_safe_point:
            if th_lastsafe:
                set_from_flat(th_lastsafe)
                logger.log("infeasible recovery mode: reverted to last safe point!")
            else:
                logger.log("alert! infeasible recovery mode failed: no last safe point to revert to.")
            wrap_up()
            continue

        loss, kl, safety, n_iter = line_search()


        if (np.isnan(loss) or np.isnan(kl) or np.isnan(safety)) or loss >= lossbefore \
            or kl >= max_kl or safety > max_sf and not accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(kl):
                logger.log("Violated because kl is NaN")
            if np.isnan(safety):
                logger.log("Violated because safety is NaN")
            if loss >= lossbefore:
                logger.log("Violated because loss not improving")
            if kl >= max_kl:
                logger.log("Violated because kl constratint is violated")
            if safety > max_sf:
                logger.log("Violated because safety constraint exceeded threshold")
            set_from_flat(thbefore)
        wrap_up()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]
