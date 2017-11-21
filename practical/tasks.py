# coding: utf-8
# This program is for the AIMS Machine Learning course. It solves tasks 1 and 2
# of practical 3.

# Task 1: run simulations to understand performance of various multi-armed
# bandit algorithms. We assume for simplicity that reward distributions are
# Bernoulli.
# We first try setting mean rewards randomly, then try the mean rewards used in
# the lower-bound construction.

# Algorithms:
# - Uniform exploration: fix a maximum time limit T, and fix a fraction of
# exploration f. Pull all arms uniformly until time f T is reached. Compute the
# mean reward for each arm. Now pull just the arm with highest mean.
# - epsilon-greedy: keep track of means of each arm (pull each arm once first,
# say). Then with probability epsilon pull a random arm, and otherwise pull the
# arm with the highest mean.
# - UCB1: keep confidence intervals for each arm. Pull arm with highest upper
# confidence bound.
# - Successive elimination: keep confidence intervals for each arm. If the upper
# confidence of an arm is less than the lower confidence of some (every other?)
# arm, then remove this arm from consideration.

import numpy as np
from bandit_algorithms import UniformExploration, EpsilonGreedy, UCB1, SuccessiveElimination, ThompsonSampling
from plot_functions import plot, plot_many

def sample_bernoulli(p):
    if np.random.rand() < p:
        return 1
    else:
        return 0

# Choose K mean rewards
def sample_mean_rewards(K):
    return np.random.rand(K)

# Lower bound mean rewards
def lower_bound_mean_rewards(K, eps=1e-1):
    a = np.random.choice(K)
    means = [0.5 for i in range(K)]
    means[a] += eps
    return means

# Computes the regret of the reward sequence, given mean rewards of each arm.
def compute_regret(rewards, mean_rewards):
    T = len(rewards)
    return T * max(mean_rewards) - np.sum(rewards)

# Computes the regret of the reward sequence, given the rewards of each arm in
# hindsight. That is, the best action (in expectation) we could have received in
# hindsight minus the reward we actually received.
def compute_regret_in_hindsight(rewards, hindsight_reward):
    return max(hindsight_reward) - np.sum(rewards)

# Takes a bandit algorithm, mean rewards and a maximum number of steps, and
# returns the sequence of rewards the bandit algorithm achieves.
def multi_armed_bandit(bandit_algorithm, mean_rewards, max_steps):
    # Use the Bernoulli distribution with means given by mean_rewards.
    rewards = []
    action_counts = {i: 0 for i in range(len(mean_rewards))}
    reward = None
    action = None
    for t in range(max_steps):
        action = bandit_algorithm.play(reward, action)
        reward = sample_bernoulli(mean_rewards[action])
        rewards.append(reward)
        action_counts[action] += 1
    return rewards, action_counts


def run_experiment(bandit_class, K, Ts, num_repeats=40, reward_generator="random"):
    cumulative_regrets = []
    for T in Ts:
        regrets = []
        for i in range(num_repeats):
            bandit_algorithm = bandit_class(K, T)
            if reward_generator == "random":
                mean_rewards = sample_mean_rewards(K)
            else:
                mean_rewards = lower_bound_mean_rewards(K)
            rewards, action_counts = multi_armed_bandit(bandit_algorithm, mean_rewards, T)
            regrets.append(compute_regret(rewards, mean_rewards))
        cumulative_regrets.append(np.mean(regrets))
        print("T: {}, cumulative regret: {}".format(T, cumulative_regrets[-1]))
    return cumulative_regrets

# Start with mean_rewards_1. At each timestep, transition with probability
# transition_probability to using the other mean reward vector, i.e. if using 1
# then switch to 2 and if using 2 then switch to 1. Then play using
# bandit_algorithm as usual.
# Takes a bandit algorithm, two sets of mean rewards and a maximum number of
# steps, and returns the sequence of rewards the bandit algorithm achieves.
# Also returns the reward for playing each arm individually -- so we can compute
# the best regret in hindsight.
def adversarial_bandit(bandit_algorithm, mean_rewards_1, mean_rewards_2,
    transition_probability, max_steps):

    # Use the Bernoulli distribution with means given by mean_rewards.
    mus = [mean_rewards_1, mean_rewards_2]
    mu_index = 0
    rewards = []
    action_counts = {i: 0 for i in range(len(mean_rewards_1))}
    reward = None
    action = None
    hindsight_reward = np.zeros(len(mean_rewards_1))
    for t in range(max_steps):
        action = bandit_algorithm.play(reward, action)
        reward = sample_bernoulli(mus[mu_index][action])
        rewards.append(reward)
        action_counts[action] += 1

        hindsight_reward += mus[mu_index]

        # Transition
        if np.random.rand() < transition_probability:
            mu_index = 1-mu_index
    return rewards, action_counts, hindsight_reward

def run_adversarial_experiment(bandit_class, K, Ts, transition_probability=0.1, num_repeats=40, reward_generator="random"):
    cumulative_regrets = []
    for T in Ts:
        regrets = []
        for i in range(num_repeats):
            bandit_algorithm = bandit_class(K, T)
            if reward_generator == "random":
                mean_rewards_1 = sample_mean_rewards(K)
                mean_rewards_2 = sample_mean_rewards(K)
            else:
                mean_rewards_1 = lower_bound_mean_rewards(K)
                mean_rewards_2 = lower_bound_mean_rewards(K)
            rewards, action_counts, hindsight_reward = adversarial_bandit(bandit_algorithm,
            mean_rewards_1, mean_rewards_2, transition_probability, T)

            regrets.append(compute_regret_in_hindsight(rewards, hindsight_reward))
        cumulative_regrets.append(np.mean(regrets))
        print("T: {}, cumulative regret: {}".format(T, cumulative_regrets[-1]))
    return cumulative_regrets

K = 10
base = 1.2
Ts = [int(20 * base**i) for i in range(30)]

algorithms = [ThompsonSampling, UniformExploration, EpsilonGreedy,
SuccessiveElimination, UCB1]
algorithm_names = ["ThompsonSampling", "UniformExploration", "EpsilonGreedy",
"SuccessiveElimination", "UCB1"]

if False:
    print("Normal bandit algorithms")
    for reward_generator in ["lower", "random"]:
        print("Reward generator: {}".format(reward_generator))
        for i in range(len(algorithms)):
            print("Algorithm: {}".format(algorithm_names[i]))
            cumulative_regrets = run_experiment(algorithms[i], K, Ts,
            reward_generator=reward_generator)
            print("Cumulative regrets: {}".format(cumulative_regrets))
            plot(np.log(Ts), np.log(cumulative_regrets),
            figname=algorithm_names[i] + "-" + reward_generator,
            xlabel="Log T", ylabel="Log cumulative regret",
            title=algorithm_names[i] + " on adversarial " + reward_generator)

# Task 2: 
print("Adversarial bandits")
for reward_generator in ["lower", "random"]:
    all_cumulative_regrets = []
    print("Reward generator: {}".format(reward_generator))
    for i in range(len(algorithms)):
        print("Algorithm: {}".format(algorithm_names[i]))
        all_cumulative_regrets.append(run_adversarial_experiment(algorithms[i], K, Ts,
        reward_generator=reward_generator))
        #print("Cumulative regrets: {}".format(cumulative_regrets))

    plot_many(np.log(Ts), all_cumulative_regrets, labels=algorithm_names,
    figname="adversarial-all" + "-" + reward_generator,
    xlabel="Log T", ylabel="Log cumulative regret",
    title="Adversarial on " + reward_generator)
