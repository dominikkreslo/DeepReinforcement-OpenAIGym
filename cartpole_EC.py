#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def build_an_episode(comments_enabled):
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        # Convert [obs] array to tensor. Call the tensor obs_v. Note that our first observation comes from the reset.
        if comments_enabled:
            print("1. Our first observation is provided as an array: \n"
              + str([obs]) + "\n")

        obs_v = torch.FloatTensor([obs])

        if comments_enabled:
            print("2. We then convert it to a tensor that we save in obs_v: \n"
                  + str(obs_v) + "\n")
        if comments_enabled:
            print("3. We then send our observation through the ANN. It returns \n"
                  + str(net(obs_v)) + "\n")
        if comments_enabled:
            print("4. We pass that value through the softmax function, where the ANN outputs then sum to 1.0. These values are \n"
                  + str(sm(net(obs_v))) + "\n")

        act_probs_v = sm(net(obs_v))

        if comments_enabled:
            print("5. We save the softmax output of the ANN as act_probs_v. This value is \n"
                  + str(act_probs_v) + "\n")

        act_probs = act_probs_v.data.numpy()[0]

        if comments_enabled:
            print("6. We get the values from this tensor and save them in act_probs. The value of act_probs is \n"
                  + str(act_probs) + "\n")

        action = np.random.choice(len(act_probs), p=act_probs)
        if comments_enabled:
            print("7. Now we want to pick an action based on the probabilities from our ANN.\n"
                  + "\t We will choose action 0 " + str(act_probs[0] * 100) + " out of every 100 selections.\n"
                  + "\t We will choose action 1 " + str(act_probs[1] * 100) + " out of every 100 selections.\n"
                  + "\t We chose action: " + str(action) + "\n")

        next_obs, reward, is_done, _ = env.step(action)

        if comments_enabled:
            print("8. Now we take this action in our environment to return a new observation, reward, and is_done.")
            print("\t Our new observation is \n"
                  + "\t \t" + str(next_obs))
            print("\t Our reward is \n"
                  + "\t \t" + str(reward))
            print("\t Are we done is \n"
                  + "\t \t" + str(is_done) + "\n")

        episode_reward += reward

        if comments_enabled:
            print("9. Our total reward for the episode is now \n "
                  + str(episode_reward) + "\n")

        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if comments_enabled:
            print("10. Add to our episode.\n"
                  + str(EpisodeStep(observation=obs, action=action)) + "\n")

        if is_done:
            if comments_enabled:
                print("Episode: " + str(Episode(reward=episode_reward, steps=episode_steps)))
            return Episode(reward=episode_reward, steps=episode_steps)

        if comments_enabled:
            print("11. Let us restart the process and get our next observation. We set obs to next_obs")
        obs = next_obs


if __name__ == "__main__":

    env = gym.make("Enduro-v0")

    #net = Net(obs_size, HIDDEN_SIZE, n_actions)
    net = Net(env.observation_space.shape[0], HIDDEN_SIZE,env.action_space.n )


    print("Cross-Entropy Algorithm (pg 81).\n")
    print("1. Algorithm: Generate N Episodes. Save them to a list.\n")

    N = 16
    EpisodeList = []
    for i in range(N):
        EpisodeList.append(build_an_episode(False))

    print("2. Algorithm: Calculate the total reward for every episode and decide on a reward boundary. "
          + "Usually, we use some percentile of all rewards, such as 50th or 70th.")


    rewards = list(map(lambda s: s.reward, EpisodeList))
    #print(rewards)
    reward_bound = np.percentile(rewards, PERCENTILE)
    #print("Reward_Bound: " + str(reward_bound))
    reward_mean = float(np.mean(rewards))
    #print("Reward_Mean:" + str(np.mean(rewards)))


    print("3. Algorithm: Throw away all episodes with a reward below the boundary.")
    EpisodeListElites = []
    for i in range(N):
        if EpisodeList[i].__getattribute__("reward") >= reward_bound:
            EpisodeListElites.append(EpisodeList[i])

    print("4. Algorithm: Train on the remaining \"elite\" episodes using observations as the input and issued actions as the desired output.")
    train_obs = []
    train_act = []
    for episode in EpisodeListElites:
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)


    #ANN setup
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    #Writer setup
    writer = SummaryWriter(comment="-cartpole")


    # Set Up Variables from our good Episodes
    obs_v = train_obs_v
    acts_v = train_act_v
    reward_b = reward_bound
    reward_m = reward_mean

    # Initialize our Artificial Neural Network. Sets the gradients for all of the parameters to zero
    optimizer.zero_grad()

    #Get ANN based scores for the actions we take by passing in observations
    action_scores_v = net(obs_v)


    # calculate the loss of the network based on
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()

    iteration_no = 0
    print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
        iteration_no, loss_v.item(), reward_m, reward_b))
    writer.add_scalar("loss", loss_v.item(), iteration_no)
    writer.add_scalar("reward_bound", reward_b, iteration_no)
    writer.add_scalar("reward_mean", reward_m, iteration_no)

    #if reward_m > 199:
     #   print("Solved!")
     #   break
    writer.close()

    print("5. Algorithm: Repeat from step 1 until we become satisfied with the result.")