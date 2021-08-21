import baseline.config as config
import numpy as np

class RandomExploreAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def selectNextAction(self):
        max_parts = config.exp['action_parts_max']
        n_parts = np.random.randint(1, max_parts + 1)
        actions = np.array([self.action_space.sample() for _ in range(n_parts)])
        if max_parts > 1:
            action = self.generateTrajectory(actions)
        else:
            action = np.squeeze(actions)
        return action, 'random'

    def generateTrajectory(self, positions):
        n_parts = len(positions)
        home = np.zeros(9)
        if config.exp['home_duration'] > 0:
            n_parts = n_parts + 1
            positions = np.vstack([home, positions])

        pos_duration = config.exp['action_size'] - config.exp['home_duration']
        xp = np.floor(np.linspace(0, pos_duration, n_parts)).astype('int')
        actions = np.vstack([np.interp(range(pos_duration), xp, positions[:, z])
                             for z in range(positions.shape[1])]).T
        actions = np.vstack([actions]+ [home] * config.exp['home_duration'])
        return actions


if __name__ == "__main__":
    import gym, real_robots
    env_id = 'REALRobot2020-{}{}{}-v0'.format('R2', 'M', '1')
    env = gym.make(env_id)
    obs_r = env.reset()

    # Configuration to repeat Round 1
    config.exp['action_parts_max'] = 1
    config.exp['home_duration'] = 0

    action_type = list(env.action_space.spaces.keys())[0]
    action_parameter_space = env.action_space[action_type]
    explorer = RandomExploreAgent(action_parameter_space)
    actions, _ = explorer.selectNextAction()
    import matplotlib.pyplot as plt
    plt.plot(actions)
    plt.show()
