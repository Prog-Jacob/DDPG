import filter_env
from ddpg import *
import gc
from gym.wrappers import RecordEpisodeStatistics

gc.enable()

ENV_NAME = 'InvertedPendulum-v4'
EPISODES = 100000
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    env = RecordEpisodeStatistics(env)
    env.reset()
    env = filter_env.makeFilteredEnv(env)
    agent = DDPG(env)
    print(env)

    for episode in range(EPISODES):
        state, info = env.reset()
        #print "episode:",episode
        # Train
        for step in range(env.spec.max_episode_steps):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if (episode % 100 == 0 and episode > 100):
            total_reward = 0
            for i in range(TEST):
                state, info = env.reset()
                for j in range(env.spec.max_episode_steps):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                ave_reward = total_reward/TEST
            print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
    env.monitor.close()

if __name__ == '__main__':
    main()
