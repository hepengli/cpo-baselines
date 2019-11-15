import time
import numpy as np
import tensorflow as tf
import functools

from baselines import logger
from baselines.common import explained_variance, zipsame, dataset, colorize
from baselines.common.cg import cg
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.input import observation_placeholder
from contextlib import contextmanager

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, ent_coef, load_path, 
                vf_stepsize, vf_iters, cg_damping, cg_iters, max_kl, max_sf):
        self.sess = sess = U.get_session()

        self.OB = OB = observation_placeholder(ob_space)
        # CREATE OUR TWO Policies
        with tf.variable_scope("pi"):
            pi = policy(observ_placeholder=OB)
        with tf.variable_scope("oldpi"):
            oldpi = policy(observ_placeholder=OB)

        # CREATE THE PLACEHOLDERS
        self.AC = AC = pi.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.R = R = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.T = T = tf.placeholder(dtype=tf.float32, shape=None)

        # Calculate the KL Divergence and Entropy
        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = ent_coef * meanent

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(pi.pd.logp(AC) - oldpi.pd.logp(AC)) # advantage * pnew / pold
        # Define loss function of the policy network
        surrlosses = - tf.reduce_mean(ratio * ADV[:,0]) - entbonus
        surrsafety = tf.reduce_mean(ratio * ADV[:,1]) * T
        loss = [surrlosses, surrsafety, meankl, entbonus, meanent]
        loss_names = ["surrlosses", "surrsafety", "meankl", "entbonus", "entropy"]
        # Define the loss function of the value network
        vferr = tf.reduce_mean(tf.square(pi.vf - R))

        # Get trainable variables
        all_var_list = get_trainable_variables("pi")
        var_list = get_pi_trainable_variables("pi")
        vf_var_list = get_vf_trainable_variables("pi")

        self.vfadam = MpiAdam(vf_var_list)
        self.vf_stepsize = vf_stepsize
        self.vf_iters = vf_iters
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.max_kl = max_kl
        self.max_sf = max_sf

        # Culculate fisher vector product
        klgrads = tf.gradients(meankl, var_list)
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

        # Get trainable variables
        self.get_flat = U.GetFlat(var_list)
        # Set trainable variables
        self.set_from_flat = U.SetFromFlat(var_list)

        # Update old plicy network
        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

        # Compute loss, safety, graident and fvp values
        self.compute_loss = U.function([OB, AC, ADV, T], loss)
        self.compute_lossesandgrad = U.function([OB, AC, ADV, T], [surrlosses, U.flatgrad(surrlosses, var_list)])
        self.compute_safetyandgrad = U.function([OB, AC, ADV, T], [surrsafety, U.flatgrad(surrsafety, var_list)])
        self.compute_vflossandgrad = U.function([OB, R], U.flatgrad(vferr, vf_var_list))
        self.compute_fvp = U.function([flat_tangent, OB, AC, ADV, T], fvp)

        self.pi = pi
        self.oldpi = oldpi
        self.step = pi.step
        self.value = pi.value
        self.initial_state = pi.initial_state

        if MPI is not None:
            nworkers = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            nworkers = 1
            rank = 0

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

        self.nworkers = nworkers
        self.rank = rank
        self.timed = timed
        self.allmean = allmean

        # Initialization
        U.initialize()
        if load_path is not None:
            pi.load(load_path)

        th_init = self.get_flat()
        if MPI is not None:
            MPI.COMM_WORLD.Bcast(th_init, root=0)

        self.set_from_flat(th_init)
        self.vfadam.sync()
        print("Init param sum", th_init.sum(), flush=True)

    def train(self, obs, returns, masks, actions, values, advs, neglogpacs, states, ep_lens, ep_rets, ep_sfts):
        # some constants
        eps = 1e-8
        backtrack_ratio = 0.8
        iters_so_far = 0

        # prepare data
        advs = (advs - advs.mean()) / (advs.std() + eps) # standardized advantage function estimate
        args = obs, actions, advs, np.mean(ep_lens)

        # compute fvp
        fvpargs = [arr[::5] for arr in args[:3]]
        def fisher_vector_product(p):
            return self.allmean(self.compute_fvp(p, *fvpargs)) + self.cg_damping * p

        # set old parameter values to new parameter values
        self.assign_old_eq_new()

        # update value network
        with self.timed("vf"):
            for _ in range(self.vf_iters):
                for (mbob, mbret) in dataset.iterbatches((obs, returns),
                include_final_partial_batch=False, batch_size=64):
                    g = self.allmean(self.compute_vflossandgrad(mbob, mbret))
                    self.vfadam.update(g, self.vf_stepsize)

        # compute loss and safety gradients
        with self.timed("computegrad"):
            lossesbefore, g = self.compute_lossesandgrad(*args)
            safetybefore, b = self.compute_safetyandgrad(*args)
        g, b = self.allmean(g), self.allmean(b)

        # compute search direction
        with self.timed("cg"):
            v = cg(fisher_vector_product, g, cg_iters=self.cg_iters, verbose=self.rank==0)
        approx_g = fisher_vector_product(v)
        q = v.dot(approx_g) # approx = g^T H^{-1} g
        delta = 2 * self.max_kl

        # residual = np.sqrt((approx_g - g).dot(approx_g - g))
        # rescale  = q / (v.dot(v))

        # ------------ Compute descent direction ------------- #
        attempt_feasible_recovery=True
        attempt_infeasible_recovery=True
        revert_to_last_safe_point=False
        linesearch_infeasible_recovery=True
        accept_violation=False

        c = np.mean(ep_sfts) - self.max_sf
        if c > 0:
            logger.log("warning! safety constraint is already violated")
        else:
            # the current parameters constitute a feasible point: save it as "last good point"
            th_lastsafe = self.get_flat()

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
            w = norm_b * cg(fisher_vector_product, unit_b, cg_iters=self.cg_iters, verbose=self.rank==0)

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
            # (this corresponds to either optim_case == 2 or 3 under certain conditions)
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
        logger.record_tabular("LagrangeLamda", lm) # dual variable for trust region
        logger.record_tabular("LagrangeNu", nu)     # dual variable for safety constraint

        if nu == 0:
            logger.log("safety constraint is not active!")

        if optim_case > 0:
            fullstep = (1. / (lm + eps) ) * ( v + nu * w )
        else:
            # current default behavior for attempting infeasible recovery:
            # take a step on natural safety gradient
            fullstep = np.sqrt(delta / (s + eps)) * w

        logger.log("descent direction computed")
        # ------------ Descent direction computed ------------- #

        thbefore = self.get_flat()

        # Safety thershold
        threshold = max(self.max_sf - np.mean(ep_sfts), 0)
        threshold = max(threshold - 1.0 * np.std(advs[:,1]), 0)

        def check_nan():
            loss, safety, kl, *_ = self.compute_loss(*args)
            if np.isnan(loss) or np.isnan(kl) or np.isnan(safety):
                logger.log("Something is NaN. Rejecting the step!")
                if np.isnan(loss):
                    logger.log("Violated because loss is NaN")
                if np.isnan(kl):
                    logger.log("Violated because kl is NaN")
                if np.isnan(safety):
                    logger.log("Violated because safety is NaN")
                self.set_from_flat(thbefore)

        def line_search(check_loss=True, check_kl=True, check_safety=True):
            loss_rejects = 0
            kl_rejects = 0
            safety_rejects  = 0
            n_iter = 0
            for n_iter, ratio in enumerate(backtrack_ratio ** np.arange(15)):
                thnew = thbefore - ratio * fullstep
                self.set_from_flat(thnew)
                loss, safety, kl, *_ = self.compute_loss(*args)
                loss_flag = loss < lossesbefore
                kl_flag = kl <= self.max_kl
                safety_flag  = safety <= threshold
                if check_loss and not(loss_flag):
                    logger.log("At backtrack itr %i, loss failed to improve." % n_iter)
                    loss_rejects += 1
                if check_kl and not(kl_flag):
                    logger.log("At backtrack itr %i, KL-Divergence violated." % n_iter)
                    logger.log("KL-Divergence violation was %.3f %%." % (100*(kl / self.max_kl) - 100))
                    kl_rejects += 1
                if check_safety and not(safety_flag):
                    logger.log("At backtrack itr %i, expression for safety constraint failed to improve." % n_iter)
                    logger.log("Safety constraint violation was %.3f %%." % (100*(safety / threshold) - 100))
                    safety_rejects += 1
                if (loss_flag or not(check_loss)) and (kl_flag or not(check_kl)) and (safety_flag or not(check_safety)):
                    logger.log("Accepted step at backtrack itr %i." % n_iter)
                    break

            if self.nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

            return loss, kl, safety, n_iter

        def wrap_up():
            loss, safety, kl, _, entropy = self.compute_loss(*args)
            logger.record_tabular("EvalLoss", loss)
            logger.record_tabular("EvalSafety", safety)
            logger.record_tabular("EvalKL", kl)
            logger.record_tabular("EvalEntropy", entropy)

        if stop_flag==True:
            wrap_up()
            return

        if optim_case == 1 and not(revert_to_last_safe_point):
            if linesearch_infeasible_recovery:
                logger.log("feasible recovery mode: constrained natural gradient step. performing linesearch on constraints.")
                loss, kl, safety, n_iter = line_search(False,True,True)
            else:
                self.set_from_flat(thbefore-fullstep)
                logger.log("feasible recovery mode: constrained natural gradient step. no linesearch performed.")
            check_nan()
            wrap_up()
            return
        elif optim_case == 0 and not(revert_to_last_safe_point):
            if linesearch_infeasible_recovery:
                logger.log("infeasible recovery mode: natural safety step. performing linesearch on constraints.")
                loss, kl, safety, n_iter = line_search(False,True,True)
            else:
                self.set_from_flat(thbefore-fullstep)
                logger.log("infeasible recovery mode: natural safety gradient step. no linesearch performed.")
            check_nan()
            wrap_up()
            return
        elif (optim_case == 0 or optim_case == 1) and revert_to_last_safe_point:
            if th_lastsafe:
                self.set_from_flat(th_lastsafe)
                logger.log("infeasible recovery mode: reverted to last safe point!")
            else:
                logger.log("alert! infeasible recovery mode failed: no last safe point to revert to.")
            wrap_up()
            return

        loss, kl, safety, n_iter = line_search()

        if (np.isnan(loss) or np.isnan(kl) or np.isnan(safety)) or loss >= lossesbefore \
            or kl >= self.max_kl or safety > self.max_sf and not accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(kl):
                logger.log("Violated because kl is NaN")
            if np.isnan(safety):
                logger.log("Violated because safety is NaN")
            if loss >= lossesbefore:
                logger.log("Violated because loss not improving")
            if kl >= self.max_kl:
                logger.log("Violated because kl constratint is violated")
            if safety > self.max_sf:
                logger.log("Violated because safety constraint exceeded threshold")
            self.set_from_flat(thbefore)
        wrap_up()


def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]
