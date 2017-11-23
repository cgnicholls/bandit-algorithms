import numpy as np
from tasks import compute_regret, compute_regret_in_hindsight
from bandit_algorithms import update_mean

def test_compute_regret(eps=1e-6):
    actions = [0,1,2]
    mean_rewards = [0.1, 0.2, 0.8]
    expected = 0.8 * 3 - 0.1 - 0.2 - 0.8
    computed = compute_regret(actions, mean_rewards)
    assert np.abs(computed - expected) < eps

    actions = [0,1,2,2,2,2,2]
    mean_rewards = [0.1, 0.2, 0.8]
    expected = 0.8 * 7 - 0.1 - 0.2 - 0.8 * 5
    computed = compute_regret(actions, mean_rewards)
    assert np.abs(computed - expected) < eps

def test_compute_hindsight_regret(eps=1e-6):
    rewards = [0.8, 0.6, 0.3]
    reward_table = np.array([[0.1, 0.3, 0.4], [0.8, 0.2, 0.1], [0.2, 0.2, 0.3],
    [0.5, 0.6, 0.7]])
    expected = 0.5+0.6+0.7 - 0.8 - 0.6 - 0.3
    computed = compute_regret_in_hindsight(rewards, reward_table)
    assert np.abs(computed - expected) < eps

def test_update_mean(eps=1e-6):
    n = 10000
    xs = np.random.randn(n+1)
    expected = np.mean(xs)
    computed = update_mean(np.mean(xs[:n]), xs[n], n)
    assert np.abs(expected - computed) < eps

def run_tests():
    test_compute_regret()
    test_compute_hindsight_regret()
    test_update_mean()

run_tests()
