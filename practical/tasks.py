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

import pickle
import numpy as np
from bandit_algorithms import UniformExploration, EpsilonGreedy, UCB1, SuccessiveElimination, ThompsonSampling, Exp4
from plot_functions import plot, plot_many, plot_from_pickle
from time import time

def sample_bernoulli(p):
    if np.random.rand() < p:
        return 1
    else:
        return 0

def sample_many_bernoulli(p):
    return (np.random.rand(len(p)) < p).astype('float32')

# Choose K mean rewards
def sample_mean_rewards(K):
    return np.random.rand(K)

# Lower bound mean rewards. Set mean reward of all arms to 0.5, and set the mean
# reward of one (randomly chosen) arm to 0.5 + epsilon.
# We have to put epsilon = Theta(sqrt(K/T)), to get the required bound on
# regret. But we cap epsilon so it is at most 0.5.
def lower_bound_mean_rewards(K, T):
    a = np.random.choice(K)
    means = [0.5 for i in range(K)]
    eps = np.sqrt(K/T)
    eps = min(0.5, eps)
    means[a] += eps
    return means

# Computes the regret of the reward sequence, given mean rewards of each arm.
def compute_regret(actions, mean_rewards):
    T = len(actions)
    algo_reward = 0.0
    for a_t in actions:
        algo_reward += mean_rewards[a_t]
    return T * max(mean_rewards) - algo_reward

# Computes the regret of the reward sequence, given the rewards of each arm in
# hindsight. That is, the best action (in expectation) we could have received in
# hindsight minus the reward we actually received.
def compute_regret_in_hindsight(rewards, reward_table):
    hindsight_reward = np.sum(reward_table, axis=1)
    return max(hindsight_reward) - np.sum(rewards)

# Takes a bandit algorithm, mean rewards and a maximum number of steps, and
# returns the sequence of rewards the bandit algorithm achieves.
def multi_armed_bandit(bandit_algorithm, mean_rewards, max_steps):
    # Use the Bernoulli distribution with means given by mean_rewards.
    rewards = []
    actions = []
    reward = None
    action = None
    for t in range(max_steps):
        action = bandit_algorithm.play(reward, action)
        reward = sample_bernoulli(mean_rewards[action])
        rewards.append(reward)
        actions.append(action)
    return rewards, actions

def run_experiment(bandit_class, K, Ts, num_repeats=40, reward_generator="random"):
    cumulative_regrets = []
    for T in Ts:
        regrets = []
        for i in range(num_repeats):
            bandit_algorithm = bandit_class(K, T)
            if reward_generator == "random":
                mean_rewards = sample_mean_rewards(K)
            else:
                mean_rewards = lower_bound_mean_rewards(K, T)
            rewards, actions = multi_armed_bandit(bandit_algorithm, mean_rewards, T)
            regrets.append(compute_regret(actions, mean_rewards))
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

    # Set up rewards and actions
    rewards = []
    K = len(mean_rewards_1)
    reward_table = np.zeros((K, max_steps))
    reward = None
    actions = []
    action = None

    # Do the loop
    for t in range(max_steps):
        # Get the action from the bandit algorithm
        action = bandit_algorithm.play(reward, action)
        actions.append(action)

        # Update the reward table (we could do this right at the start, since
        # it's an oblivious adversary).
        reward_table[:,t] = sample_many_bernoulli(mus[mu_index])
        reward = reward_table[action,t]
        rewards.append(reward)

        # Transition
        if np.random.rand() < transition_probability:
            mu_index = 1-mu_index
            
    return reward_table, rewards, actions

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
                mean_rewards_1 = lower_bound_mean_rewards(K, T)
                mean_rewards_2 = lower_bound_mean_rewards(K, T)
            reward_table, rewards, actions = adversarial_bandit(bandit_algorithm,
            mean_rewards_1, mean_rewards_2, transition_probability, T)

            regrets.append(compute_regret_in_hindsight(rewards, reward_table))
        cumulative_regrets.append(np.mean(regrets))
        print("T: {}, cumulative regret: {}".format(T, cumulative_regrets[-1]))
    return cumulative_regrets

if __name__ == "__main__":
    K = 10
    base = 1.2
    num_Ts = 45
    Ts = [int(20 * base**i) for i in range(num_Ts)]
    print("Ts: {}".format(Ts))

    run_standard = True
    run_adversarial = True
    reward_generators = ["lower"]

    algorithms = [SuccessiveElimination, UCB1, ThompsonSampling,
    UniformExploration, EpsilonGreedy, Exp4]
    algorithm_names = ["SuccessiveElimination", "UCB1", "ThompsonSampling",
    "UniformExploration", "EpsilonGreedy", "Exp4"]

    #algorithms = [Exp4]
    #algorithm_names = ["Exp4"]

    #algorithms = [EpsilonGreedy]
    #algorithm_names = ["EpsilonGreedy"]

    if run_standard:
        print("Standard bandits")
        for reward_generator in reward_generators:
            start_time = time()
            all_cumulative_regrets = []
            print("Reward generator: {}".format(reward_generator))
            for i in range(len(algorithms)):
                print("Algorithm: {}".format(algorithm_names[i]))
                all_cumulative_regrets.append(run_experiment(algorithms[i], K, Ts,
                reward_generator=reward_generator))
                #print("Cumulative regrets: {}".format(cumulative_regrets))

            # Save to file.
            figname = "standard-all" + "-" + reward_generator + "-numTs-" + str(num_Ts)
            pickle_name = figname + ".pickle"
            pickle_out = open(pickle_name,"wb")
            result_dict = {"xs": np.log(Ts),
                           "ys": np.log(all_cumulative_regrets),
                           "labels": algorithm_names,
                           "xlabel": "Log T",
                           "ylabel": "Log cumulative regret",
                           "title": "Standard bandits on " + reward_generator}
            pickle.dump(result_dict, pickle_out)
            pickle_out.close()

            plot_from_pickle(pickle_name, figname)
            print("Time taken to test all algorithms: {}".format(time() - start_time))

    # Task 2: 
    if run_adversarial:
        print("Adversarial bandits")
        for reward_generator in reward_generators:
            start_time = time()
            all_cumulative_regrets = []
            print("Reward generator: {}".format(reward_generator))
            for i in range(len(algorithms)):
                print("Algorithm: {}".format(algorithm_names[i]))
                all_cumulative_regrets.append(run_adversarial_experiment(algorithms[i], K, Ts,
                reward_generator=reward_generator))
                #print("Cumulative regrets: {}".format(cumulative_regrets))

            # Save to file.
            figname = "adversarial-all" + "-" + reward_generator + "-numTs-" + str(num_Ts)
            pickle_name = figname + ".pickle"
            pickle_out = open(pickle_name,"wb")
            result_dict = {"xs": np.log(Ts),
                           "ys": np.log(all_cumulative_regrets),
                           "labels": algorithm_names,
                           "xlabel": "Log T",
                           "ylabel": "Log cumulative regret",
                           "title": "Adversarial bandits on " + reward_generator}
            pickle.dump(result_dict, pickle_out)
            pickle_out.close()

            plot_from_pickle(pickle_name, figname)
            print("Time taken to test all algorithms: {}".format(time() - start_time))
