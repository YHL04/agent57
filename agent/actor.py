

import time

from .replay_buffer import LocalBuffer
from environment import Env


class Actor:
    """
    Class to be asynchronously run by Learner, use self.run() for main training
    loop. This class creates a local buffer to store data before sending completed
    Episode to Learner through rpc. All communication is through numpy array.

    Parameters:
    learner_rref (RRef): Learner RRef to reference the learner
    """

    def __init__(self, learner_rref, id, env_name):
        self.learner_rref = learner_rref
        self.id = id

        self.env = Env(env_name)

        self.local_buffer = LocalBuffer()

    def get_action(self, obs, state):
        """
        Uses learner RRef and rpc async to call queue_request to get action
        from learner.

        Parameters:
        obs (np.array): frames with shape (batch_size, n_channels, h, w)
        state (np.array): recurrent states with shape (batch_size, state_len, d_model)

        Returns:
        Future() object that when used with .wait(), halts until value is ready from
        the learner. It returns action(float) and state(np.array)

        """
        return self.learner_rref.rpc_async().queue_request(obs, state)

    def return_episode(self, episode):
        """
        Once episode is completed return_episode uses learner_rref and rpc_async
        to call return_episode to return Episode object to learner for training.

        Parameters:
        episode (Episode)

        Returns:
        future_await (Future): halts with .wait() until learner is finished
        """
        return self.learner_rref.rpc_async().return_episode(episode)

    def run(self):
        """
        Main actor training loop, calls queue_request to get action and
        return_episode to return finished episode
        """

        while True:
            obs = self.env.reset()
            state = None
            action, state = self.get_action(obs, state).wait()

            start = time.time()
            total_reward = 0
            done = False

            while not done:
                action, next_state = self.get_action(obs, state).wait()
                next_obs, reward, done = self.env.step(action)

                self.local_buffer.add(obs, action, reward, state)

                obs = next_obs
                state = next_state

                total_reward += reward

            episode = self.local_buffer.finish(total_reward, time.time()-start)
            self.return_episode(episode).wait()

            self.env.render_episode()
