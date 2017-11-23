import numpy as np

# Given the current mean, a new element to add to the list, and the size of the
# current list, computes the mean of the new list.
def update_mean(current_mean, new_x, n):
    return (current_mean * float(n) + new_x) / float(n+1)

# A bandit algorithm takes in a number of arms, and every time it is called has
# to choose which arm to pull.
class BanditAlgorithm:
    def __init__(self, K):
        self.K = K

    # Decide which arm to play, given the previous reward and previous action.
    # We set previous_reward and previous_action to None if it's the first go.
    def play(self, previous_reward, previous_action):
        pass

class UniformExploration(BanditAlgorithm):
    def __init__(self, K, T):
        self.K = K
        self.set_exploration_steps(T)
        self.t = 0
        self.rewards = {i: [] for i in range(self.K)}
        self.mean_rewards = None
        self.best_arm = None

    def play(self, previous_reward, previous_action):
        # Add the previous rewards
        if self.t > 0:
            self.rewards[previous_action].append(previous_reward)

        # Compute means and let best_arm be the arm with highest mean
        if self.t == self.exploration_steps:
            self.mean_rewards = {i : np.mean(reward) for i, reward in self.rewards.iteritems()}
            self.best_arm = max(self.mean_rewards, key=self.mean_rewards.get)

        # Either explore uniformly, or pick the arm with highest mean reward
        if self.t >= self.exploration_steps:
            action = self.best_arm
        else:
            action = self.t % self.K
        self.t += 1
        return action

    def set_exploration_steps(self, T):
        self.exploration_steps = int((T / self.K)**0.667 * np.log(T))

class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, K, T):
        self.K = K
        self.t = 0
        self.epsilon = self.compute_epsilon(0)
        self.rewards = {i: [] for i in range(self.K)}
        self.mean_rewards = {i: 0.0 for i in range(self.K)}

    # Computes epsilon as t^(-1/3) * (K * log t)^(1/3).
    def compute_epsilon(self, t):
        if t <= 1:
            return 1.0
        else:
            return np.min([t**(-0.33) * (self.K * np.log(t))**(0.33), 1.0])

    def play(self, previous_reward, previous_action):
        # Update epsilon
        self.epsilon = self.compute_epsilon(self.t)

        # Update mean rewards
        if not previous_reward is None:
            self.rewards[previous_action].append(float(previous_reward))
            # The only reward to be updated is the mean reward for the previous
            # action
            # New fast way of doing it: only update the mean reward for the
            # previous reward, and update it in constant time.
            self.mean_rewards[previous_action] = update_mean(self.mean_rewards[previous_action], previous_reward, len(self.rewards[previous_action]))
            # Old way of doing it:
            #self.mean_rewards[previous_action] = np.mean(self.rewards[previous_action])
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.K)
        else:
            action = max(self.mean_rewards, key=self.mean_rewards.get)

        self.t += 1
        return action

class UCB1(BanditAlgorithm):
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.t = 0
        self.rewards = {i: [] for i in range(self.K)}
        self.mean_rewards = {i: 0.0 for i in range(self.K)}
        self.confidences = {i: 0.0 for i in range(self.K)}

    # Computes rt(a) = sqrt(2 * log T / nt(a)), where nt(a) is the number of
    # times action a has been played.
    def compute_rt(self, T, n):
        return np.sqrt(2 * np.log(T) / n)

    def play(self, previous_reward, previous_action):
        # Compute mean rewards and confidence intervals
        if not previous_reward is None:
            self.rewards[previous_action].append(float(previous_reward))
            # Only have to update the mean reward and confidence of the
            # previous_action.
            self.mean_rewards[previous_action] = update_mean(self.mean_rewards[previous_action], previous_reward, len(self.rewards[previous_action]))
            self.confidences[previous_action] = self.compute_rt(self.T, len(self.rewards[previous_action]))
            
            # Old way:
            #self.mean_rewards = {k : np.mean(v) if len(v) > 0 else 0.0 for k,v in self.rewards.iteritems()}
            #self.confidences = {k : self.compute_rt(self.T, len(v)) if len(v) > 0 else 0.0 for k,v in self.rewards.iteritems()}

        # Play each arm at least once, then play the arm with highest upper
        # confidence.
        if self.t < self.K:
            action = self.t
        else:
            upper_confidences = {k : self.mean_rewards[k] + self.confidences[k] for k,v in self.mean_rewards.iteritems()}
            action = max(upper_confidences, key=upper_confidences.get)

        self.t += 1
        return action

class SuccessiveElimination(BanditAlgorithm):
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.t = 0
        self.rewards = {i: [] for i in range(self.K)}
        self.mean_rewards = {i: 0.0 for i in range(self.K)}
        self.confidences = {i: 0.0 for i in range(self.K)}
        self.active_arms = range(self.K)

        self.arms_to_play_this_round = [i for i in self.active_arms]

    # Computes rt(a) = sqrt(2 * log T / nt(a)), where nt(a) is the number of
    # times action a has been played.
    def compute_rt(self, T, n):
        return np.sqrt(2 * np.log(T) / n)

    # Each round we first play all the arms to play, until self.arms_to_play has
    # length 0. Then we remove arms and reset self.arms_to_play.
    def play(self, previous_reward, previous_action):
        # Compute mean rewards and confidence intervals
        if not previous_reward is None:
            self.rewards[previous_action].append(float(previous_reward))
            # Only have to update mean rewards and confidences of previous_action.
            self.mean_rewards[previous_action] = update_mean(self.mean_rewards[previous_action], previous_reward, len(self.rewards[previous_action]))
            self.confidences[previous_action] = self.compute_rt(self.T, len(self.rewards[previous_action]))

            # Old way:
            #self.mean_rewards = {k : np.mean(v) if len(v) > 0 else 0.0 for k,v in self.rewards.iteritems()}
            #self.confidences = {k : self.compute_rt(self.T, len(v)) if len(v) > 0 else 0.0 for k,v in self.rewards.iteritems()}

        if len(self.arms_to_play_this_round) == 0:
            # If we played all the active arms, then we compute upper and lower
            # confidence bounds, and eliminate any arm that is dominated by some
            # other arm. In practice, we just compute the maximum lowest
            # confidence bound and eliminate any arms whose upper confidence
            # bound is smaller than it.
            upper_confidences = {k : self.mean_rewards[k] + self.confidences[k] for k,v in self.mean_rewards.iteritems()}
            lower_confidences = {k : self.mean_rewards[k] - self.confidences[k] for k,v in self.mean_rewards.iteritems()}
            max_lower = max(lower_confidences.values())
            new_active_arms = []
            for arm in self.active_arms:
                if upper_confidences[arm] >= max_lower:
                    new_active_arms.append(arm)
            self.active_arms = new_active_arms
            self.arms_to_play_this_round = [i for i in self.active_arms]

        # Now choose the next arm to play
        action = self.arms_to_play_this_round.pop()
        self.t += 1
        return action

class ThompsonSampling(BanditAlgorithm):
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.t = 0
        # The arm history stores a vector for each arm. The first element
        # is the number of 0s the arm has output and the second is the number of
        # 1s the arm has output.
        self.arm_history = {i: [1,1] for i in range(self.K)}

    def play(self, previous_reward, previous_action):
        # First update the arm history
        if not previous_action is None:
            self.arm_history[previous_action][previous_reward] += 1

        # Now sample a mean vector from the posterior distribution
        mu_sample = [np.random.beta(self.arm_history[a][1],
        self.arm_history[a][0]) for a in range(self.K)]
        
        # Pick the action with largest mean in the sample
        action = max(range(self.K), key=lambda x: mu_sample[x])
        return action

# Exp4 is designed to have good regret bounds in the adversarial bandit
# situation. We implement this with cost notation, where cost = 1 - reward. The
# interface is as for rewards though, since we make the adjustment inside 'play'.
class Exp4(BanditAlgorithm):
    def __init__(self, K, T, epsilon=0.3, gamma=0.3):
        self.K = K
        self.T = T
        self.t = 0
        self.weights = np.array([1 for i in range(self.K)])
        self.gamma = 1 / (4 * self.T) # gamma should be in [0, 1/2T)
        U = self.K / (1 - self.gamma)
        self.epsilon = np.sqrt(np.log(self.K) / (3*U)) # epsilon should equal sqrt(log K / (3*U))

        # Initialise fake costs to 0.
        self.fake_costs = np.array([0.0 for i in range(self.K)])
        self.picked_expert = None
        self.prob_dist = np.array([1.0 / self.K for i in range(self.K)])

    def update_weights(self, weights, fake_costs, epsilon):
        return weights * (1.0 - epsilon)**fake_costs

    def play(self, previous_reward, previous_action):
        # previous_cost is the cost ct(at) for the previous action
        # (previous_action).
        # If this is the first action, then just choose randomly.
        if previous_reward is None:
            self.picked_expert = np.random.choice(self.K)
            return self.picked_expert

        previous_cost = 1.0 - previous_reward

        # Define fake costs
        self.fake_costs = np.zeros(self.K)
        if (not self.picked_expert is None) and (self.picked_expert == previous_action):
            self.fake_costs[self.picked_expert] = float(previous_cost) / self.prob_dist[self.picked_expert]

        # Then update weights
        self.weights = self.update_weights(self.weights, self.fake_costs, self.epsilon)

        # Then compute prob dist.
        self.prob_dist = self.weights / np.sum(self.weights)

        # Draw an expert from the prob dist.
        self.picked_expert = np.random.choice(self.K, p=self.prob_dist)
        # With probability gamma, we choose randomly
        if np.random.rand() < self.gamma:
            return np.random.choice(self.K)
        else:
            return self.picked_expert
