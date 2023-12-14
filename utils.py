import gym
import torch
import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from deepq_network import CNN, MLPNet

import torch.nn.functional as F


def discretize_state(state):
    if (type(state) is tuple):
        state = state[0]
    discrete_state = (min(2, max(-2, int((state[0]) / 0.05))), \
                        min(2, max(-2, int((state[1]) / 0.1))), \
                        min(2, max(-2, int((state[2]) / 0.1))), \
                        min(2, max(-2, int((state[3]) / 0.1))), \
                        min(2, max(-2, int((state[4]) / 0.1))), \
                        min(2, max(-2, int((state[5]) / 0.1))), \
                        int(state[6]), \
                        int(state[7]))

    return discrete_state


def epsilon_greedy(q_func, state, eps, env_actions):
    prob = np.random.random()

    if prob < eps:
        return random.choice(range(env_actions))
    elif isinstance(q_func, CNN) or isinstance(q_func, MLPNet):
        with torch.no_grad():
            return q_func(state).max(1)[1].item()
    else:
        qvals = [q_func[state + (action, )] for action in range(env_actions)]
        return np.argmax(qvals)

def boltzmann_exploration(q_func, state, env_actions):
    if isinstance(q_func, CNN) or isinstance(q_func, MLPNet):
        with torch.no_grad():
            output_action = q_func(state)

            # print("boltzmann output_action.size():", output_action.size())# torch.Size([1, 4])
            prob = F.softmax(output_action,dim=1)

            # print("boltzmann output_action.prob():", prob.size())# torch.Size([1, 4])

            prob = torch.squeeze(prob)

            # print("squeeze boltzmann output_action.prob():", prob.size())# torch.Size([4])
            
            # norm_output_action = torch.exp(output_action)/torch.sum(torch.exp(output_action))
            n = 1
            replace = True
            action = prob.multinomial(num_samples=n, replacement=replace)[0]
            # action = norm_output_action.multinomial(num_samples=n, replacement=replace)[0]
            """
            print("output_action")
            print(output_action)
            print("norm_output_action")
            print(norm_output_action)
            print("boltzmann_exploration action")
            print(action)
            # action = np.random.choice(output_action.shape[0], 1, p=norm_output_action)
            """
            return action.item()
    else:
        qvals = [q_func[state + (action, )] for action in range(env_actions)]
        # print("boltzmann_exploration qvals----------------------")
        # print(qvals)
        
        qvals = np.exp(qvals)/sum(np.exp(qvals))
        # print("softmax boltzmann_exploration qvals----------------------")
        # print(qvals)
        # print("boltzmann_exploration else quit--------------------------------------")
        # quit()

        return np.argmax(qvals)

def greedy(qstates_dict, state, env_actions):
    qvals = [qstates_dict[state + (action, )] for action in range(env_actions)]
    return max(qvals)


def discounted_return(episode_return, gamma):
    g = 0
    for i, r in enumerate(episode_return):
        g += gamma**i * r

    return g


def decay_epsilon(curr_eps, exploration_final_eps):
    if curr_eps < exploration_final_eps:
        return curr_eps
    
    return curr_eps * 0.996


def image_input(obs):
    if (type(obs) is tuple):
        obs = obs[0]
    net_input = np.expand_dims(obs, 0)
    net_input = torch.from_numpy(net_input)

    return net_input


def build_qnetwork(env_actions, learning_rate, input_shape, network, device):
    if network == 'cnn':
        qnet = CNN(env_actions)
    else:
        # model = 'linear'
        qnet = MLPNet(input_shape, env_actions)
    return qnet.to(device), torch.optim.RMSprop(qnet.parameters(), lr=learning_rate)


def fit(qnet, qnet_optim, qtarget_net, loss_func, \
        frames, actions, rewards, next_frames, dones, \
        gamma, env_actions, device):

    frames_t = torch.cat(frames).to(device)
    actions = torch.tensor(actions, device=device)
    q_t = qnet(frames_t) # q_t tensor has shape (batch, env_actions)
    q_t_selected = torch.sum(q_t * torch.nn.functional.one_hot(actions, env_actions), 1) 

    dones = torch.tensor(dones, device=device)
    rewards = torch.tensor(rewards, device=device)
    frames_tp1 = torch.cat(next_frames).to(device)
    q_tp1_best = qtarget_net(frames_tp1).max(1)[0].detach() 
    ones = torch.ones(dones.size(-1), device=device)
    q_tp1_best = (ones - dones) * q_tp1_best
    q_targets = rewards + gamma * q_tp1_best

    # td error
    loss = loss_func(q_t_selected, q_targets)
    qnet_optim.zero_grad()
    loss.backward()
    qnet_optim.step()


def update_target_network(qnet, qtarget_net):
    qtarget_net.load_state_dict(qnet.state_dict())

"""#original
def save_model(qnet, episode, path):
    torch.save(qnet.state_dict(), os.path.join(path, 'qnetwork_{}.pt'.format(episode)))
"""
def save_specific_model(qnet, episode, path, specific):
    torch.save(qnet.state_dict(), os.path.join(path, specific + '_qnetwork_{}.pt'.format(episode)))

def plot_rewards(chosen_agents, agents_returns, num_episodes, window):
    num_intervals = int(num_episodes / window)
    for agent, agent_total_returns in zip(chosen_agents, agents_returns):
        print(len(agent_total_returns))
        print("\n{} lander average reward = {}".format(agent, sum(agent_total_returns) / num_episodes))
        l = []
        for j in range(num_intervals):
            l.append(round(np.mean(agent_total_returns[j * 100 : (j + 1) * 100]), 1))
        plt.plot(range(0, num_episodes, window), l)

    plt.xlabel("Episodes")
    plt.ylabel("Reward per {} episodes".format(window))
    plt.title("RL Lander(s)")
    plt.legend(chosen_agents, loc="lower right")
    plt.show()
