import numpy as np
from baselines.common.runners import AbstractEnvRunner
import time

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self, op_model):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos, ep_obs, ep_values = [], [ob for ob in self.obs.copy()], []

        if self.model.name_scope == 'rg':
            op_actions, op_values, _, _ = op_model.step(self.obs)

        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            if self.model.name_scope == 'em':
                op_actions = np.zeros(shape=[self.env.num_envs, 1])
                op_values = op_model.pi.q_value(self.obs)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions, op_actions, op_values)

            if self.model.name_scope == 'rg':
                if np.all(self.dones):
                    op_actions, op_values, _, _ = op_model.step(self.obs)
                    rewards += op_values[:,0]

            safety = []
            for n, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                safety.append(info.get('s'))
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
                    ep_values.append(info.get('reg'))
                    ep_obs.append(self.obs[n])
            mb_rewards.append(np.stack([rewards, safety], axis=1))

        ep_obs[-self.env.num_envs:] = []
        ep_obs = np.asarray(ep_obs, dtype=self.obs.dtype)
        ep_values = np.asarray(ep_values, dtype=self.obs.dtype)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal[:,None] - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal[:,None] * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_advs, mb_neglogpacs)),
            mb_states, epinfos, ep_obs, ep_values)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])