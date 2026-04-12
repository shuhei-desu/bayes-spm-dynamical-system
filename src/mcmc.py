import numpy as np
from scipy.special import gammaln

from free_energy import free_energy_replica


NEG_LARGE = -1e100


# =========================================================
# Proposal updates
# =========================================================

def update_cij(cij: np.ndarray, num_replica: int, n_osci: int):
    """
    Flip one off-diagonal entry c_ij independently for each replica.

    Parameters
    ----------
    cij : ndarray of shape (R, N, N)
        Binary pairwise interaction indicators.
    num_replica : int
        Number of replicas.
    n_osci : int
        Number of oscillators.

    Returns
    -------
    cij_new : ndarray of shape (R, N, N)
        Proposed updated tensor.
    proposed_mask : ndarray of shape (R, N, N)
        Binary mask indicating which entry was proposed in each replica.
    """
    cij_new = cij.copy()
    replica_idx = np.arange(num_replica)

    row = np.random.randint(n_osci, size=num_replica)
    col = np.random.randint(n_osci - 1, size=num_replica)
    col += (col >= row)  # skip diagonal

    cij_new[replica_idx, row, col] = 1.0 - cij_new[replica_idx, row, col]

    proposed_mask = np.zeros_like(cij, dtype=int)
    proposed_mask[replica_idx, row, col] = 1
    return cij_new, proposed_mask


def update_cijk(cijk: np.ndarray, num_replica: int, n_osci: int):
    """
    Flip one ordered triplet entry c_ijk independently for each replica,
    with i, j, k all distinct.

    Parameters
    ----------
    cijk : ndarray of shape (R, N, N, N)
        Binary triplet interaction indicators.
    num_replica : int
        Number of replicas.
    n_osci : int
        Number of oscillators.

    Returns
    -------
    cijk_new : ndarray of shape (R, N, N, N)
        Proposed updated tensor.
    proposed_mask : ndarray of shape (R, N, N, N)
        Binary mask indicating which entry was proposed in each replica.
    """
    if n_osci < 3:
        raise ValueError("n_osci must be at least 3.")

    cijk_new = cijk.copy()
    replica_idx = np.arange(num_replica)

    i = np.random.randint(n_osci, size=num_replica)

    j_raw = np.random.randint(n_osci - 1, size=num_replica)
    j = j_raw + (j_raw >= i)

    k_raw = np.random.randint(n_osci - 2, size=num_replica)
    aij = np.minimum(i, j)
    bij = np.maximum(i, j)
    k = k_raw + (k_raw >= aij) + (k_raw >= (bij - 1))

    cijk_new[replica_idx, i, j, k] = 1.0 - cijk_new[replica_idx, i, j, k]

    proposed_mask = np.zeros_like(cijk, dtype=int)
    proposed_mask[replica_idx, i, j, k] = 1
    return cijk_new, proposed_mask


def update_cijk_symmetric(
    cijk_s: np.ndarray,
    num_replica: int,
    n_osci: int,
    J_s,
    K_s,
):
    """
    Flip one unordered pair (j, k) with j < k for each chosen i and replica.

    Parameters
    ----------
    cijk_s : ndarray of shape (R, N, N, N)
        Binary symmetric triplet indicators.
    num_replica : int
        Number of replicas.
    n_osci : int
        Number of oscillators.
    J_s, K_s : list of ndarray
        Precomputed symmetric index pairs for each i.

    Returns
    -------
    cijk_s_new : ndarray of shape (R, N, N, N)
        Proposed updated tensor.
    proposed_mask : ndarray of shape (R, N, N, N)
        Binary mask indicating which entry was proposed in each replica.
    """
    if n_osci < 3:
        raise ValueError("n_osci must be at least 3.")

    cijk_s_new = cijk_s.copy()
    replica_idx = np.arange(num_replica)

    i = np.random.randint(n_osci, size=num_replica)

    lengths = np.fromiter((len(J_s[ii]) for ii in i), dtype=int)
    if np.any(lengths == 0):
        raise ValueError("No symmetric pairs available for some i.")

    t = (np.random.rand(num_replica) * lengths).astype(int)
    j = np.array([J_s[ii][tt] for ii, tt in zip(i, t)], dtype=int)
    k = np.array([K_s[ii][tt] for ii, tt in zip(i, t)], dtype=int)

    cijk_s_new[replica_idx, i, j, k] = 1.0 - cijk_s_new[replica_idx, i, j, k]

    proposed_mask = np.zeros_like(cijk_s, dtype=int)
    proposed_mask[replica_idx, i, j, k] = 1
    return cijk_s_new, proposed_mask


def update_tau(tau_array: np.ndarray, step_width_tau: np.ndarray):
    """
    Perturb one tau entry per replica with a uniform random step.
    """
    num_replica, d1, d2 = tau_array.shape
    proposed_mask = np.zeros((num_replica, d1, d2))

    d1_idx = np.random.randint(d1, size=num_replica)
    d2_idx = np.random.randint(d2, size=num_replica)
    proposed_mask[np.arange(num_replica), d1_idx, d2_idx] = 1.0

    tau_array_new = (
        tau_array.copy()
        + step_width_tau
        * np.random.uniform(-1, 1, size=(num_replica, d1, d2))
        * proposed_mask
    )
    return tau_array_new, proposed_mask


def update_log_xi(
    log_xi_array: np.ndarray,
    num_replica: int,
    n_osci: int,
    step_width_xi: np.ndarray,
):
    """
    Perturb one log-xi entry per replica with a uniform random step.
    """
    proposed_mask = np.zeros((num_replica, n_osci))
    random_indices = np.random.randint(n_osci, size=num_replica)
    np.put_along_axis(proposed_mask, random_indices[:, np.newaxis], 1, axis=1)

    log_xi_array_new = (
        log_xi_array.copy()
        + step_width_xi
        * np.random.uniform(-1, 1, size=(num_replica, n_osci))
        * proposed_mask
    )
    return log_xi_array_new, proposed_mask


def update_L(L_array: np.ndarray, num_replica: int):
    """
    Propose L -> L +/- 1 independently for each replica.
    """
    return L_array.copy() + 2 * np.random.randint(0, 2, num_replica) - 1


def update_d(d_array: np.ndarray, num_replica: int):
    """
    Flip one binary basis-selection entry per replica.
    """
    d_array_new = d_array.copy()
    flip_indices = np.random.randint(0, d_array.shape[1], size=num_replica)

    d_array_new[np.arange(num_replica), flip_indices] = (
        1 - d_array_new[np.arange(num_replica), flip_indices]
    )

    proposed_mask = np.zeros_like(d_array, dtype=int)
    proposed_mask[np.arange(num_replica), flip_indices] = 1
    return d_array_new, proposed_mask


# =========================================================
# Log priors
# =========================================================

def log_prior_cij(cij: np.ndarray):
    return np.sum(np.log(0.5) * np.ones_like(cij), axis=(1, 2)).reshape(-1, 1)


def log_prior_cijk(cijk: np.ndarray):
    return np.sum(np.log(0.5) * np.ones_like(cijk), axis=(1, 2, 3)).reshape(-1, 1)


def log_prior_uniform(x_array: np.ndarray, lower: float, upper: float):
    """
    Uniform log prior for arrays of shape (R, D).
    """
    inside = (x_array >= lower) & (x_array <= upper)
    log_prob = np.full_like(x_array, NEG_LARGE, dtype=float)
    log_prob[inside] = -np.log(upper - lower)
    return np.sum(log_prob, axis=1, keepdims=True)


def log_prior_uniform_2(x_array: np.ndarray, lower: float, upper: float):
    """
    Uniform log prior for arrays of shape (R, D1, D2).
    """
    inside = (x_array >= lower) & (x_array <= upper)
    log_prob = np.full_like(x_array, NEG_LARGE, dtype=float)
    log_prob[inside] = -np.log(upper - lower)
    return np.sum(log_prob, axis=(1, 2), keepdims=True).reshape(-1, 1)


def log_prior_gamma(x_array: np.ndarray, shape: float, scale: float):
    """
    Gamma log prior for positive arrays of shape (R, D).

    Note
    ----
    Currently unused, but kept for completeness.
    """
    a = shape
    theta = scale

    log_prob = np.full_like(x_array, NEG_LARGE, dtype=float)
    mask = x_array > 0
    log_prob[mask] = (
        -gammaln(a)
        - a * np.log(theta)
        + (a - 1) * np.log(x_array[mask])
        - x_array[mask] / theta
    )
    return np.sum(log_prob, axis=1, keepdims=True)


def log_prior_L(L_array: np.ndarray, L_min: int, L_max: int):
    """
    Discrete uniform prior over integers in [L_min, L_max].
    """
    L_array = np.asarray(L_array)
    log_prob = np.full_like(L_array, NEG_LARGE, dtype=np.float64)

    mask = (L_array >= L_min) & (L_array <= L_max)
    log_prob[mask] = -np.log(L_max - L_min + 1)
    return log_prob.reshape(-1, 1)


def log_prior_d(d_array: np.ndarray):
    return np.sum(np.log(0.5) * np.ones_like(d_array), axis=1).reshape(-1, 1)


# =========================================================
# Metropolis helpers
# =========================================================

def _select_accepted_state(
    current_state: np.ndarray,
    proposed_state: np.ndarray,
    F_old: np.ndarray,
    F_new: np.ndarray,
    log_accept_ratio: np.ndarray,
):
    """
    Accept or reject proposals independently for each replica.

    Parameters
    ----------
    current_state : ndarray
        Current state.
    proposed_state : ndarray
        Proposed state.
    F_old, F_new : ndarray of shape (R, 1)
        Old and proposed free energies.
    log_accept_ratio : ndarray of shape (R, 1)
        Log acceptance ratio.

    Returns
    -------
    next_state : ndarray
        Accepted/rejected state after Metropolis step.
    F_next : ndarray of shape (R, 1)
        Accepted/rejected free energies.
    accept_mask : ndarray of shape (R, 1)
        Binary acceptance mask.
    """
    log_uniform = np.log(np.random.uniform(0.0, 1.0, size=(log_accept_ratio.shape[0], 1)))
    accept_mask = (log_accept_ratio > log_uniform).astype(float)
    reject_mask = 1.0 - accept_mask

    reshape_dims = (accept_mask.shape[0],) + (1,) * (current_state.ndim - 1)
    accept_broadcast = accept_mask.reshape(reshape_dims)
    reject_broadcast = reject_mask.reshape(reshape_dims)

    next_state = accept_broadcast * proposed_state + reject_broadcast * current_state
    F_next = accept_mask * F_new + reject_mask * F_old
    return next_state, F_next, accept_mask


def _compute_free_energy(
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    delta_t,
    M,
    N_osci,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    """
    Wrapper around free_energy_replica to keep the update kernels readable.

    Notes
    -----
    The current free_energy_replica signature does NOT take num_replica or M.
    num_replica is inferred from cij.shape[0].
    """
    return free_energy_replica(
        delta_phi,
        G_L,
        cij,
        cijk_a,
        cijk_s,
        tau_array * tau_array,
        np.exp(log_xi_array),
        L2_array,
        L3_array,
        d_array,
        delta_t,
        N_osci,
        L2_min,
        L2_max,
        L3_min,
        L3_max,
    )


def _metropolis_update(
    *,
    current_state,
    propose_fn,
    log_prior_fn,
    F_old,
    beta,
    free_energy_fn,
    accepted_mask_ndim=None,
):
    proposal_out = propose_fn(current_state)

    if isinstance(proposal_out, tuple):
        proposed_state, proposed_mask = proposal_out
    else:
        proposed_state = proposal_out
        proposed_mask = None

    F_new = free_energy_fn(proposed_state)


    F_old = np.asarray(F_old).reshape(-1, 1)
    F_new = np.asarray(F_new).reshape(-1, 1)
    beta = np.asarray(beta).reshape(-1, 1)


    log_accept_ratio = (
        (F_old - F_new) * beta
        + log_prior_fn(proposed_state)
        - log_prior_fn(current_state)
    )

    next_state, F_next, accept_mask = _select_accepted_state(
        current_state=current_state,
        proposed_state=proposed_state,
        F_old=F_old,
        F_new=F_new,
        log_accept_ratio=log_accept_ratio,
    )

    if proposed_mask is None:
        return next_state, F_next.ravel(), accept_mask.ravel()

    if accepted_mask_ndim is None:
        return next_state, F_next.ravel(), proposed_mask, accept_mask

    reshape_shape = (-1,) + (1,) * accepted_mask_ndim
    accepted_mask = proposed_mask * accept_mask.reshape(reshape_shape)
    return next_state, F_next.ravel(), proposed_mask, accepted_mask


# =========================================================
# Single-step Metropolis updates
# =========================================================

def ms_cij(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    return _metropolis_update(
        current_state=cij,
        propose_fn=lambda x: update_cij(x, num_replica, N_osci),
        log_prior_fn=log_prior_cij,
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            proposed,
            cijk_a,
            cijk_s,
            tau_array,
            log_xi_array,
            L2_array,
            L3_array,
            d_array,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=2,
    )


def ms_cijk_a(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    return _metropolis_update(
        current_state=cijk_a,
        propose_fn=lambda x: update_cijk(x, num_replica, N_osci),
        log_prior_fn=log_prior_cijk,
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            cij,
            proposed,
            cijk_s,
            tau_array,
            log_xi_array,
            L2_array,
            L3_array,
            d_array,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=3,
    )


def ms_cijk_s(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
    J_s,
    K_s,
):
    return _metropolis_update(
        current_state=cijk_s,
        propose_fn=lambda x: update_cijk_symmetric(x, num_replica, N_osci, J_s, K_s),
        log_prior_fn=log_prior_cijk,
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            cij,
            cijk_a,
            proposed,
            tau_array,
            log_xi_array,
            L2_array,
            L3_array,
            d_array,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=3,
    )


def ms_tau(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    return _metropolis_update(
        current_state=tau_array,
        propose_fn=lambda x: update_tau(x, step_width_tau),
        log_prior_fn=lambda x: log_prior_uniform_2(x, lower=0.01, upper=10.0),
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            cij,
            cijk_a,
            cijk_s,
            proposed,
            log_xi_array,
            L2_array,
            L3_array,
            d_array,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=2,
    )


def ms_xi(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    return _metropolis_update(
        current_state=log_xi_array,
        propose_fn=lambda x: update_log_xi(x, num_replica, N_osci, step_width_xi),
        log_prior_fn=lambda x: log_prior_uniform(
            np.exp(x), lower=0.03, upper=1500.0
        ),
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            cij,
            cijk_a,
            cijk_s,
            tau_array,
            proposed,
            L2_array,
            L3_array,
            d_array,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=1,
    )


def ms_L2(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    L2_next, F_next, accept_mask = _metropolis_update(
        current_state=L2_array,
        propose_fn=lambda x: update_L(x, num_replica),
        log_prior_fn=lambda x: log_prior_L(x, L_min=1, L_max=3),
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            cij,
            cijk_a,
            cijk_s,
            tau_array,
            log_xi_array,
            proposed,
            L3_array,
            d_array,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=None,
    )
    return L2_next.astype(int), F_next, accept_mask


def ms_L3(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    L3_next, F_next, accept_mask = _metropolis_update(
        current_state=L3_array,
        propose_fn=lambda x: update_L(x, num_replica),
        log_prior_fn=lambda x: log_prior_L(x, L_min=1, L_max=3),
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            cij,
            cijk_a,
            cijk_s,
            tau_array,
            log_xi_array,
            L2_array,
            proposed,
            d_array,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=None,
    )
    return L3_next.astype(int), F_next, accept_mask


def ms_d(
    F_old,
    num_replica,
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_array,
    log_xi_array,
    L2_array,
    L3_array,
    d_array,
    step_width_tau,
    step_width_xi,
    delta_t,
    M,
    N_osci,
    beta,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    return _metropolis_update(
        current_state=d_array,
        propose_fn=lambda x: update_d(x, num_replica),
        log_prior_fn=log_prior_d,
        F_old=F_old,
        beta=beta,
        free_energy_fn=lambda proposed: _compute_free_energy(
            num_replica,
            delta_phi,
            G_L,
            cij,
            cijk_a,
            cijk_s,
            tau_array,
            log_xi_array,
            L2_array,
            L3_array,
            proposed,
            delta_t,
            M,
            N_osci,
            L2_min,
            L2_max,
            L3_min,
            L3_max,
        ),
        accepted_mask_ndim=1,
    )

