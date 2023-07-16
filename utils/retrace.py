

import torch


def compute_off_policy_returns(q_t, v_t, r_t, discount_t, c_t):
    """
    Calculates targets. v_t is just exp_q_t
    """
    g = r_t[-1] + discount_t[-1] * v_t[-1]
    returns = [g]
    for t in reversed(range(q_t.size(0))):
        g = r_t[t] + discount_t[t] * (v_t[t] - c_t[t] * q_t[t] + c_t[t] * g)
        returns.insert(0, g)

    return torch.stack(returns, dim=0).detach()


def compute_retrace_target(q_t, a_t, r_t, discount_t, c_t, pi_t):
    """
    Args:
        q_t (T, B, action_dim): expected q value at time t
        a_t (T, B): action index at time t
        r_t (T, B): rewards at time t
        discount_t (T, B): discount at time t
        c_t (T-1, B): importance weights at time t
        pi_t (T, B, action_dim): target policy probs at time t
    """
    exp_q_t = (pi_t * q_t).sum(axis=-1)
    q_a_t = q_t[a_t]

    return compute_off_policy_returns(q_a_t, exp_q_t, r_t, discount_t, c_t)


def compute_retrace_loss(q_t, q_t1, a_t, a_t1, r_t, pi_t, mu_t, discount_t, retrace_lambda=0.95, eps=1e-8):
    """

    Args:
        q_t (T, B, action_dim): expected q values at time t
        q_t1 (T, B, action_dim): target q values at time t+1
        a_t (T, B): actions at time t
        a_t1 (T, B): actions at time t+1
        r_t (T, B): rewards at time t
        pi_t (T, B, action_dim): online model action probs
        mu_t (T, B, action_dim): target model action probs
        gamma (int): discount factor
        retrace_lambda (int=0.95): lambda constant for retrace loss
        eps (int=1e-2): small value to add to mu for numerical stability
    """

    with torch.no_grad():
        # get probability of a_t at time t
        pi_a_t = pi_t[a_t]
        mu_a_t = mu_t[a_t]

        # compute cutting trace coefficients in retrace
        c_t = torch.minimum(torch.tensor(1.0), pi_a_t / (mu_a_t + eps)) * retrace_lambda

        # get retrace targets
        target = compute_retrace_target(q_t, a_t, r_t, discount_t, c_t, pi_t)

    # get expected q value of taking action a_t
    expected = q_t[a_t]

    loss = (target - expected) ** 2
    return loss


if __name__ == "__main__":
    T, B, action_dim = 3, 2, 4

    x = torch.ones((T, B, action_dim))
    y = torch.ones((T, B))

    loss = compute_retrace_loss(x, x, y, y, y, x, x, y)
