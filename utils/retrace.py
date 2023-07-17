

import torch

from value_rescale import value_rescaling, inverse_value_rescaling


def get_index(x, idx):
    """
    Code modified from
    https://github.com/michaelnny/deep_rl_zoo/blob/main/deep_rl_zoo/nonlinear_bellman.py
    https://github.com/deepmind/trfl/blob/master/trfl/retrace_ops.py

    Returns x that correspond to index in idx

    Args:
        x (T, B, action_dim): tensor to be indexed
        idx (T, B): indices

    Returns:
        indexed (T, B): indexed tensor
    """
    T, B, action_dim = x.shape
    assert idx.shape == (T, B)

    x = x.view(-1, action_dim)
    idx = idx.view(-1)

    indexed = x[torch.arange(x.size(0)), idx]
    indexed = indexed.view(T, B)

    return indexed


def compute_retrace_target(q_t, a_t, r_t, discount_t, c_t, pi_t):
    """
    Args:
        q_t (T, B, action_dim): target network q value at time t
        a_t (T, B): action index at time t
        r_t (T, B): rewards at time t
        discount_t (T, B): discount at time t
        c_t (T-1, B): importance weights at time t
        pi_t (T, B, action_dim): target policy probs at time t
    """
    exp_q_t = (pi_t * q_t).sum(axis=-1)
    q_a_t = get_index(q_t, a_t)

    q_a_t = q_a_t[:-1, ...]
    c_t = c_t[:-1, ...]

    g = r_t[-1] + discount_t[-1] * exp_q_t[-1]
    returns = [g]
    for t in reversed(range(q_t.size(0))):
        g = r_t[t] + discount_t[t] * (exp_q_t[t] - c_t[t] * q_a_t[t] + c_t[t] * g)
        # g = r_t[t] + discount_t[t] * exp_q_t[t] + discount_t[t] * c_t[t] * (g - q_a_t[t])
        returns.insert(0, g)

    return torch.stack(returns, dim=0).detach()


def compute_retrace_loss(q_t,
                         q_t1,
                         a_t,
                         a_t1,
                         r_t,
                         pi_t1,
                         mu_t1,
                         discount_t,
                         retrace_lambda=0.95,
                         eps=1e-8
                         ):
    """

    Args:
        q_t (T, B, action_dim): expected q values at time t
        q_t1 (T, B, action_dim): target q values at time t+1
        a_t (T, B): actions at time t
        a_t1 (T, B): actions at time t+1
        r_t (T, B): rewards at time t
        pi_t1 (T, B, action_dim): online model action probs at time t+1
        mu_t1 (T, B, action_dim): target model action probs at time t+1
        discount_t (T, B): discount factor
        retrace_lambda (int=0.95): lambda constant for retrace loss
        eps (int=1e-2): small value to add to mu for numerical stability
    """
    T, B, action_dim = q_t.shape

    assert q_t.shape == (T, B, action_dim)
    assert q_t1.shape == (T, B, action_dim)
    assert a_t.shape == (T, B)
    assert a_t1.shape == (T, B)
    assert r_t.shape == (T, B)
    assert pi_t1.shape == (T, B, action_dim)
    assert mu_t1.shape == (T, B, action_dim)
    assert discount_t.shape == (T, B)

    with torch.no_grad():
        # get probability of a_t at time t
        pi_a_t1 = get_index(pi_t1, a_t)

        # compute cutting trace coefficients in retrace
        c_t1 = torch.minimum(torch.tensor(1.0), pi_a_t1 / (mu_t1 + eps)) * retrace_lambda

        # get transformed retrace targets
        q_t1 = inverse_value_rescaling(q_t1)
        target = compute_retrace_target(q_t1, a_t1, r_t, discount_t, c_t1, pi_t1)
        target = value_rescaling(target)

    # get expected q value of taking action a_t
    expected = get_index(q_t, a_t)

    loss = (target - expected) ** 2
    return loss


if __name__ == "__main__":
    T, B, action_dim = 3, 2, 4

    x = torch.ones((T, B, action_dim))
    y = torch.ones((T, B))

    loss = compute_retrace_loss(x, x, y, y, y, x, x, y)

