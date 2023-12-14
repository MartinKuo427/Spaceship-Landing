import gym
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import *

import collections


class ClassReplayMemory(object):
    def __init__(self, capacity):
        self.transitions = []
        self.max_capacity = capacity
        self.next_transition_index = 0


    def length(self):
        return len(self.transitions)


    def record(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.next_transition_index >= self.length():
            self.transitions.append(transition)
        else:
            self.transitions[self.next_transition_index] = transition   # overwrite old experiences

        self.next_transition_index = (self.next_transition_index + 1) % self.max_capacity


    def minibatch_sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            transition_index = random.randint(0, self.length() - 1)
            transition = self.transitions[transition_index]
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones


class MLPNet(nn.Module):
    def __init__(self, input_shape, env_actions):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, env_actions)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return self.out(x)

def random_lander(environment, number_episodes, max_steps_per_episodes):# , print_freq=500, render_freq=500

    print_frequency = 500
    render_frequency = 500
    is_render = False

    return_per_episode = [0.0]

    # open log file
    # martinc
    reward_log_file = open('./logs/random_lander_log.csv', 'w')
    reward_log_file.write("episode,mean_reward\n")

    for episode in range(number_episodes):
        state = environment.reset()
        if episode % render_frequency == 0:
            is_render = True
        else:
            is_render = False

        for step in range(max_steps_per_episodes):
            if is_render:
                environment.render()
            action = environment.action_space.sample()
            observation, reward, done, _, _ = environment.step(action)
            return_per_episode[-1] += reward
    
            if done:
                if episode % print_frequency == 0:
                    print("Episode:", episode, " finished after steps:", step, " Total return:", return_per_episode[-1])

                # martinc
                if (episode + 1) % 100 == 0:
                    mean_100ep_reward = round(np.mean(return_per_episode[-101:-1]), 1)
                    print("\nMLPNet_epsilon_greedy Last 100 episodes mean reward: {}".format(mean_100ep_reward))
                    reward_log_file.write("%d,%f\n" %(episode, mean_100ep_reward))
                    reward_log_file.flush()


                return_per_episode.append(0.0)
                break
    
            state = observation

    reward_log_file.close()
    return return_per_episode




"""
# def deepqn_lander(environment, number_episodes, max_steps_per_episodes):
def deepqn_lander(env, n_episodes, gamma, lr, min_eps, max_steps_per_episodes,\
                batch_size=32, memory_capacity=50000, \
                network='linear', learning_starts=1000, \
                train_freq=1, target_network_update_freq=1000, \
                print_freq=500, render_freq=500, save_freq=1000):
"""
def deepqn_lander(environment, number_episodes, max_steps_per_episodes):
    gamma = 0.99
    lr = 1e-4
    min_eps = 1e-2
    batch_size = 32
    memory_capacity = 50000
    learning_starts = 1000
    train_freq = 1
    target_network_update_freq = 1000
    print_frequency = 500
    render_frequency = 500
    save_freq = 1000
    is_render = False

    # open log file
    reward_log_file = open('./logs/deepqn_lander_log.csv', 'w')
    reward_log_file.write("episode,mean_reward\n")
    
    # set device to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # path to save checkpoints
    PATH = "./models"
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    num_actions = environment.action_space.n
    input_shape = environment.observation_space.shape[-1]
    # qnet, qnet_optim = build_qnetwork(num_actions, lr, input_shape, network, device)
    # qtarget_net, _ = build_qnetwork(num_actions, lr, input_shape, network, device)


    number_actions = environment.action_space.n
    input_shape = environment.observation_space.shape[-1]

    qnet = MLPNet(input_shape, number_actions).to(device)
    qtarget_net = MLPNet(input_shape, number_actions).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(qnet.parameters(), lr=lr)


    
    qtarget_net.load_state_dict(qnet.state_dict())
    qnet.train()
    qtarget_net.eval()
    reply_memory_object = ClassReplayMemory(memory_capacity)

    epsilon = 1.0 
    return_per_ep = [0.0] 
    saved_mean_reward = None
    t = 0

    for i in range(number_episodes):
        curr_state = image_input(environment.reset())
        if (i + 1) % render_frequency == 0:
            render = True
        else:
            render = False

        each_episodes_time_step = 0
        while True:
            # print("t:", t)
            if render:
                environment.render()


            action = epsilon_greedy(qnet, curr_state.to(device), epsilon, num_actions)

            next_state, reward, done, _, _ = environment.step(action)

            next_state = image_input(next_state)

            reply_memory_object.record(curr_state, action, float(reward), next_state, float(done))


            if t > learning_starts and t % train_freq == 0:
                states, actions, rewards, next_states, dones = reply_memory_object.minibatch_sample(batch_size)
                fit(qnet, optimizer, qtarget_net, loss_fn, states, actions, rewards, next_states, dones, gamma, num_actions, device)

            if t > learning_starts and t % target_network_update_freq == 0:
                update_target_network(qnet, qtarget_net)

            t += 1
            each_episodes_time_step += 1
            return_per_ep[-1] += reward

            if (done or (each_episodes_time_step >= max_steps_per_episodes)):
                print("each_episodes_time_step:", each_episodes_time_step)
                if (i + 1) % print_frequency == 0:
                    print("\nEpisode: {}".format(i + 1))
                    print("Episode return : {}".format(return_per_ep[-1]))
                    print("Total time-steps: {}".format(t))

                if (i + 1) % 100 == 0:
                    mean_100ep_reward = round(np.mean(return_per_ep[-101:-1]), 1)
                    print("\nMLPNet_epsilon_greedy Last 100 episodes mean reward: {}".format(mean_100ep_reward))
                    reward_log_file.write("%d,%f\n" %(i, mean_100ep_reward))
                    reward_log_file.flush()

                if t > learning_starts and (i + 1) % save_freq == 0:
                    if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                        print("\nSaving MLPNet_epsilon_greedy model due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                        save_specific_model(qnet, i + 1, PATH, "MLPNet_epsilon_greedy")
                        saved_mean_reward = mean_100ep_reward

                return_per_ep.append(0.0)
                epsilon = decay_epsilon(epsilon, min_eps)

                break

            curr_state = next_state

    reward_log_file.close()
    return return_per_ep


def dqn_boltzmann_exploration_lander(environment, number_episodes, max_steps_per_episodes):
    gamma = 0.99
    lr = 1e-4
    min_eps = 1e-2
    batch_size = 32
    memory_capacity = 50000
    learning_starts = 1000
    train_freq = 1
    target_network_update_freq = 1000
    print_frequency = 500
    render_frequency = 500
    save_freq = 1000
    is_render = False

    # open log file
    reward_log_file = open('./logs/dqn_boltzmann_exploration_lander_log.csv', 'w')
    reward_log_file.write("episode,mean_reward\n")
    
    # set device to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # path to save checkpoints
    PATH = "./models"
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    num_actions = environment.action_space.n
    input_shape = environment.observation_space.shape[-1]


    number_actions = environment.action_space.n
    input_shape = environment.observation_space.shape[-1]

    qnet = MLPNet(input_shape, number_actions).to(device)
    qtarget_net = MLPNet(input_shape, number_actions).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(qnet.parameters(), lr=lr)


    
    qtarget_net.load_state_dict(qnet.state_dict())
    qnet.train()
    qtarget_net.eval()
    reply_memory_object = ClassReplayMemory(memory_capacity)

    epsilon = 1.0 
    return_per_ep = [0.0] 
    saved_mean_reward = None
    t = 0

    for i in range(number_episodes):
        curr_state = image_input(environment.reset())
        if (i + 1) % render_frequency == 0:
            render = True
        else:
            render = False

        each_episodes_time_step = 0
        while True:
            # print("t:", t)
            if render:
                environment.render()

            action = boltzmann_exploration(qnet, curr_state.to(device), num_actions)

            next_state, reward, done, _, _ = environment.step(action)

            next_state = image_input(next_state)

            reply_memory_object.record(curr_state, action, float(reward), next_state, float(done))

            if t > learning_starts and t % train_freq == 0:
                states, actions, rewards, next_states, dones = reply_memory_object.minibatch_sample(batch_size)
                fit(qnet, optimizer, qtarget_net, loss_fn, states, actions, rewards, next_states, dones, gamma, num_actions, device)

            if t > learning_starts and t % target_network_update_freq == 0:
                update_target_network(qnet, qtarget_net)

            t += 1
            each_episodes_time_step += 1
            return_per_ep[-1] += reward

            if (done or (each_episodes_time_step >= max_steps_per_episodes)):
                # print("each_episodes_time_step:", each_episodes_time_step)
                if (i + 1) % print_frequency == 0:
                    print("\nEpisode: {}".format(i + 1))
                    print("Episode return : {}".format(return_per_ep[-1]))
                    print("Total time-steps: {}".format(t))

                if (i + 1) % 100 == 0:
                    mean_100ep_reward = round(np.mean(return_per_ep[-101:-1]), 1)
                    print("\nMLPNet_dqn_boltzmann_exploration_lander Last 100 episodes mean reward: {}".format(mean_100ep_reward))
                    reward_log_file.write("%d,%f\n" %(i, mean_100ep_reward))
                    reward_log_file.flush()

                if t > learning_starts and (i + 1) % save_freq == 0:
                    if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                        print("\nSaving MLPNet_dqn_boltzmann_exploration_lander model due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                        save_specific_model(qnet, i + 1, PATH, "MLPNet_dqn_boltzmann_exploration_lander")
                        saved_mean_reward = mean_100ep_reward

                return_per_ep.append(0.0)
                epsilon = decay_epsilon(epsilon, min_eps)

                break

            curr_state = next_state

    reward_log_file.close()
    return return_per_ep


# def qlearning_lander(env, n_episodes, gamma, lr, min_eps, max_steps_per_episodes, print_freq=500, render_freq=500):
def qlearning_lander(environment, number_episodes, max_steps_per_episodes):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    return_per_ep = [0.0]
    epsilon = 1.0
    num_actions = environment.action_space.n

    gamma = 0.99
    lr = 1e-1
    min_eps = 1e-2
    batch_size = 32
    memory_capacity = 50000
    learning_starts = 1000
    train_freq = 1
    target_network_update_freq = 1000
    print_frequency = 500
    render_frequency = 500
    save_freq = 1000
    is_render = False

    # open log file
    reward_log_file = open('./logs/qlearning_lander_log.csv', 'w')
    reward_log_file.write("episode,mean_reward\n")

    for i in range(number_episodes):
        t = 0
        if (i + 1) % render_frequency == 0:
            render = True
        else:
            render = False

        curr_state = discretize_state(environment.reset())
        
        each_episodes_time_step = 0
        while True:
            if render:
                environment.render()

            action = epsilon_greedy(q_states, curr_state, epsilon, num_actions)

            qstate = curr_state + (action, )

            observation, reward, done, _, _ = environment.step(action)
            next_state = discretize_state(observation)

            if not done:
                q_states[qstate] += lr * (reward + gamma * greedy(q_states, next_state, num_actions) - q_states[qstate]) # (S', A') non terminal state
            else:
                q_states[qstate] += lr * (reward - q_states[qstate])    

            return_per_ep[-1] += reward

            if (done or (each_episodes_time_step >= max_steps_per_episodes)):
                if (i + 1) % print_frequency == 0:
                    print("\nEpisode finished after {} timesteps".format(t + 1))
                    print("Episode {}: Total Return = {}".format(i + 1, return_per_ep[-1]))
                    print("Total keys in q_states dictionary = {}".format(len(q_states)))

                if (i + 1) % 100 == 0:
                    mean_100ep_reward = round(np.mean(return_per_ep[-101:-1]), 1)
                    print("qlearning Last 100 episodes mean reward: {}".format(mean_100ep_reward))
                    reward_log_file.write("%d,%f\n" %(i, mean_100ep_reward))
                    reward_log_file.flush()

                epsilon = decay_epsilon(epsilon, min_eps)
                return_per_ep.append(0.0)

                break

            curr_state = next_state
            t += 1
            each_episodes_time_step += 1

    reward_log_file.close()
    return return_per_ep




def qlearning_boltzmann_exploration_lander(environment, number_episodes, max_steps_per_episodes):
    q_states = collections.defaultdict(float)   # note that the first insertion of a key initializes its value to 0.0
    return_per_ep = [0.0]
    epsilon = 1.0
    num_actions = environment.action_space.n

    gamma = 0.99
    lr = 1e-1
    min_eps = 1e-2
    batch_size = 32
    memory_capacity = 50000
    learning_starts = 1000
    train_freq = 1
    target_network_update_freq = 1000
    print_frequency = 500
    render_frequency = 500
    save_freq = 1000
    is_render = False

    # open log file
    reward_log_file = open('./logs/qlearning_boltzmann_exploration_lander_log.csv', 'w')
    reward_log_file.write("episode,mean_reward\n")

    for i in range(number_episodes):
        t = 0
        if (i + 1) % render_frequency == 0:
            render = True
        else:
            render = False

        curr_state = discretize_state(environment.reset())
        
        each_episodes_time_step = 0
        while True:
            if render:
                environment.render()

            action = epsilon_greedy(q_states, curr_state, epsilon, num_actions)

            qstate = curr_state + (action, )

            observation, reward, done, _, _ = environment.step(action)
            next_state = discretize_state(observation)

            if not done:
                q_states[qstate] += lr * (reward + gamma * boltzmann_exploration(q_states, next_state, num_actions) - q_states[qstate])
            else:
                q_states[qstate] += lr * (reward - q_states[qstate])
            return_per_ep[-1] += reward

            if (done or (each_episodes_time_step >= max_steps_per_episodes)):
                if (i + 1) % print_frequency == 0:
                    print("\nEpisode finished after {} timesteps".format(t + 1))
                    print("Episode {}: Total Return = {}".format(i + 1, return_per_ep[-1]))
                    print("Total keys in q_states dictionary = {}".format(len(q_states)))

                if (i + 1) % 100 == 0:
                    mean_100ep_reward = round(np.mean(return_per_ep[-101:-1]), 1)
                    print("qlearning_boltzmann_exploration Last 100 episodes mean reward: {}".format(mean_100ep_reward))
                    reward_log_file.write("%d,%f\n" %(i, mean_100ep_reward))
                    reward_log_file.flush()

                epsilon = decay_epsilon(epsilon, min_eps)
                return_per_ep.append(0.0)

                break

            curr_state = next_state
            t += 1
            each_episodes_time_step += 1

    reward_log_file.close()
    return return_per_ep



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', nargs='+', help='algorithm(s) of choice to train lander')
    parser.add_argument('--number_episodes', type=int, help='number_episodes', default=100, required=False)                            # default number of episodes is 10000
    parser.add_argument('--max_steps_per_episodes', type=int, help='max_steps_per_episodes', default=1000000000000, required=False)                            # default number of episodes is 10000
    parser.add_argument('--lr', type=float, help='step-size (or learning rate) used in sarsa, q-learning, dqn', default=1e-3, required=False)   # default step-size is 0.001
    parser.add_argument('--gamma', type=float, help='discount rate, should be 0 < gamma < 1', default=0.99, required=False)                     # default gamma is 0.99
    parser.add_argument('--final_eps', type=float, help='decay epsilon unti it reaches its \'final_eps\' value', default=1e-2, required=False)  # default final eploration epsilon is 0.01
    args = parser.parse_args()

    environment = gym.make("LunarLander-v2")

    print("args.agents:", args.agents)
    if args.agents[0] == "random":
        print("\n random agent")
        t_rewards = random_lander(environment, args.number_episodes, args.max_steps_per_episodes)
        # print("random agent total rewards:", t_rewards)
    elif args.agents[0] == "deepqn":
        print("\n deepqn agent")
        # t_rewards = dqn_lander(environment, args.number_episodes, args.gamma, args.lr, args.final_eps, args.max_steps_per_episodes)
        # t_rewards = deepqn_lander(environment, args.number_episodes, args.max_steps_per_episodes)
        # t_rewards = deepqn_lander(environment, args.number_episodes, args.gamma, args.lr, args.final_eps, args.max_steps_per_episodes)
        t_rewards = deepqn_lander(environment, args.number_episodes, args.max_steps_per_episodes)
        # print("deepqn agent total rewards:", t_rewards)
    
    elif args.agents[0] == "dqn_bolt":
        print("\n dqn_bolt agent ...")
        # total_rewards = dqn_boltzmann_exploration_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)
        total_rewards = dqn_boltzmann_exploration_lander(environment, args.number_episodes, args.max_steps_per_episodes)
    
    elif args.agents[0] == "qlearning":# qlearning_lander
        print("\n qlearning agent") # def qlearning_lander(env, n_episodes, gamma, lr, min_eps, max_steps_per_episodes, print_freq=500, render_freq=500):
        t_rewards = qlearning_lander(environment, args.number_episodes, args.max_steps_per_episodes)
        # print("deepqn agent total rewards:", t_rewards)

    elif args.agents[0] == "qlearning_bolt":# qlearning_lander
        print("\n qlearning_bolt agent") # def qlearning_lander(env, n_episodes, gamma, lr, min_eps, max_steps_per_episodes, print_freq=500, render_freq=500):
        t_rewards = qlearning_boltzmann_exploration_lander(environment, args.number_episodes, args.max_steps_per_episodes)
        # print("deepqn agent total rewards:", t_rewards)


    environment.close()


if __name__ == '__main__':
    main()



