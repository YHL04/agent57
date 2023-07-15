

import torch


def compute_retrace_target(q_t, a_t, r_t, gamma, c_t, pi_t):
    """
    Args:
        q_t (torch.tensor): expected q value at time t [T, B, action_dim]

    """
    exp_q_t = (pi_t * q_t).sum(axis=-1)

    return exp_q_t


def compute_retrace_loss(q_t, q_t1, a_t, a_t1, r_t, pi_t, mu_t, gamma, retrace_lambda, eps=1e-8):
    """
    All variables are shaped (T, B)

    Args:
        q_t (torch.tensor): expected q values at time t [T, B, action_dim]
        q_t1 (torch.tensor): target q values at time t+1 [T, B, action_dim]
        a_t (torch.tensor): actions at time t [T, B]
        a_t1 (torch.tensor): actions at time t+1 [T, B]
        r_t (torch.tensor): rewards at time t [T, B]
        pi_t (torch.tensor): online model action probs [T, B, action_dim]
        mu_t (torch.tensor): target model action probs [T, B, action_dim]
        gamma (int): discount factor
        retrace_lambda (int): lambda constant for retrace loss
        eps (int): small value to add to mu for numerical stability
    """

    with torch.no_grad():
        # get probability of a_t at time t
        pi_a_t = pi_t[a_t]
        mu_a_t = mu_t[a_t]

        # compute cutting trace coefficients in retrace
        c_t = torch.minimum(torch.tensor(1.0), pi_a_t / (mu_a_t + eps)) * retrace_lambda

        # get retrace targets
        target = compute_retrace_target(q_t, a_t, r_t, gamma, c_t, pi_t)

    # get expected q value of taking action a_t
    expected = q_t[a_t]

    td_error = target - expected
    loss = 0.5 * td_error**2

    return loss
