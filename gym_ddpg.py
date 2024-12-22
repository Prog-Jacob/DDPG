import filter_env
from ddpg import *
import gc
from gym.wrappers import Monitor

gc.enable()

ENV_NAME = 'InvertedPendulum-v2'
EPISODES = 100000
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    env = Monitor(env, "./monitor", force=True)  # Path to save videos/logs
    env = filter_env.makeFilteredEnv(env)
    agent = DDPG(env)
    print(env)
    env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in range(env.spec.timestep_limit):
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
                state = env.reset()
                for j in range(env.spec.timestep_limit):
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
