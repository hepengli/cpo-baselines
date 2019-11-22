import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym


class Q(object):
    """
    Encapsulates fields and methods for RL policy and Q-function estimation with shared parameters
    """

    def __init__(self, observations, actions, latent, sess=None, **tensors):
        """
        Parameters:
        ----------

        observations    tensorflow placeholder in which the observations will be fed

        actions         tensorflow placeholder in which the actions will be fed

        latent          latent state from which policy distribution parameters should be inferred

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.OB = observations
        self.AC = actions
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        latent = tf.layers.flatten(latent)

        self.sess = sess or tf.get_default_session()
        self.q = fc(latent, 'q', 2)

    def _evaluate(self, variables, observation, action, **extra_feed):
        sess = self.sess
        feed_dict = {self.OB: adjust_shape(self.OB, observation), self.AC: adjust_shape(self.AC, action)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def value(self, ob, ac, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.q, ob, ac, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_Q_fn(env, Q_network, normalize_observations=False, **network_kwargs):
    if isinstance(Q_network, str):
        network_type = Q_network
        Q_network = get_network_builder(network_type)(**network_kwargs)

    def Q_function(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, action_placeholder=None):
        ob_space = env.observation_space
        ac_space = env.action_space
        OB = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch, name='OB')
        AC = action_placeholder if action_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch, name='AC')

        extra_tensors = {}

        if normalize_observations and OB.dtype == tf.float32 and AC.dtype == tf.float32:
            encoded_ob, ob_rms = _normalize_clip_observation(OB)
            encoded_ac, ac_rms = _normalize_clip_observation(AC)
            extra_tensors['ob_rms'] = ob_rms
            extra_tensors['ac_rms'] = ac_rms
        else:
            encoded_ob = OB
            encoded_ac = AC

        encoded_ob = encode_observation(ob_space, encoded_ob)
        encoded_ac = encode_observation(ac_space, encoded_ac)
        encoded_x = tf.concat([encoded_ob, encoded_ac])

        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            latent = Q_network(encoded_x)
            if isinstance(latent, tuple):
                latent, recurrent_tensors = latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    latent, recurrent_tensors = Q_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)

        qf = Q(
            observations=OB,
            actions=AC,
            latent=latent,
            sess=sess,
            **extra_tensors
        )
        return qf

    return Q_function


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms