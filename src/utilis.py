import numpy as np
from scipy.linalg import cho_factor, cho_solve


def compute_phase_increments(phi_traj, time_traj):
    """
    Compute phase increments and the sampling interval from a trajectory.

    Parameters
    ----------
    phi_traj : ndarray of shape (T, N)
        Phase trajectory.
    time_traj : ndarray of shape (T,)
        Time stamps. Assumed to be uniformly spaced.

    Returns
    -------
    delta_phi : ndarray of shape (T-1, N)
        Consecutive phase differences.
    delta_t : float
        Time step size.
    """
    phi_traj = np.asarray(phi_traj)
    time_traj = np.asarray(time_traj)

    if phi_traj.shape[0] != time_traj.shape[0]:
        raise ValueError("phi_traj and time_traj must have the same length.")
    if len(time_traj) < 2:
        raise ValueError("time_traj must contain at least two time points.")

    delta_phi = phi_traj[1:, :] - phi_traj[:-1, :]
    delta_t = time_traj[1] - time_traj[0]
    return delta_phi, delta_t


def build_feature_matrix_per_oscillator(phis_all, L2_max, L3_max):
    """
    Build regression feature matrices for each oscillator.

    For each oscillator k, this function constructs a design matrix whose columns
    contain:
    - a constant term,
    - pairwise sin/cos harmonics up to order L2_max,
    - two types of three-body sin/cos harmonics up to order L3_max.

    Parameters
    ----------
    phis_all : ndarray of shape (T, N)
        Observed phases.
    L2_max : int
        Maximum harmonic order for pairwise terms.
    L3_max : int
        Maximum harmonic order for three-body terms.

    Returns
    -------
    G_list : ndarray of shape (N, T-1, n_features)
        Feature matrices for each oscillator.
    """
    n_time, n_osc = phis_all.shape
    n_samples = n_time - 1
    phis = phis_all[:-1, :]

    n_features = 1 + 2 * n_osc * L2_max + 4 * n_osc * n_osc * L3_max
    G_list = np.zeros((n_osc, n_samples, n_features))

    three_body_diff_a = (
        2 * phis.reshape(n_samples, 1, 1, n_osc)
        - phis.reshape(n_samples, n_osc, 1, 1)
        - phis.reshape(n_samples, 1, n_osc, 1)
    )
    three_body_diff_a = three_body_diff_a.transpose(0, 1, 3, 2).reshape(
        n_samples, n_osc, n_osc * n_osc
    )

    three_body_diff_s = (
        phis.reshape(n_samples, 1, n_osc, 1)
        + phis.reshape(n_samples, 1, 1, n_osc)
        - 2 * phis.reshape(n_samples, n_osc, 1, 1)
    )
    three_body_diff_s = three_body_diff_s.transpose(0, 1, 3, 2).reshape(
        n_samples, n_osc, n_osc * n_osc
    )

    for k in range(n_osc):
        pairwise_diff = phis - phis[:, [k]]
        features = [np.ones((n_samples, 1))]

        for l in range(1, L2_max + 1):
            features.append(np.sin(l * pairwise_diff))
        for l in range(1, L2_max + 1):
            features.append(np.cos(l * pairwise_diff))

        for l in range(1, L3_max + 1):
            features.append(np.sin(l * three_body_diff_a[:, k, :]))
        for l in range(1, L3_max + 1):
            features.append(np.cos(l * three_body_diff_a[:, k, :]))

        for l in range(1, L3_max + 1):
            features.append(np.sin(l * three_body_diff_s[:, k, :]))
        for l in range(1, L3_max + 1):
            features.append(np.cos(l * three_body_diff_s[:, k, :]))

        G_list[k, :, :] = np.concatenate(features, axis=1)

    return G_list


def quad_form_via_cholesky(K, y):
    """
    Compute y^T K^{-1} y using a Cholesky factorization.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Symmetric positive-definite matrix.
    y : ndarray of shape (n,) or (n, 1)
        Vector in the quadratic form.

    Returns
    -------
    float
        The scalar value y^T K^{-1} y.
    """
    c, lower = cho_factor(K, lower=True)
    alpha = cho_solve((c, lower), y)
    return float(y.T @ alpha)


def precompute_symmetric_pairs(N):
    """
    For each index i, enumerate all unordered pairs (j, k) with j < k and j, k != i.

    Parameters
    ----------
    N : int
        Number of oscillators.

    Returns
    -------
    J_s : list of ndarray
        J_s[i] contains the first indices j for oscillator i.
    K_s : list of ndarray
        K_s[i] contains the second indices k for oscillator i.
    """
    J_s, K_s = [], []
    for i in range(N):
        others = np.array([r for r in range(N) if r != i])
        jj, kk = np.triu_indices(len(others), k=1)
        J_s.append(others[jj])
        K_s.append(others[kk])
    return J_s, K_s


def compute_step_sizes(C_stepsize_list, d_stepsize_list, num_replica, param_dim, beta, num_data):
    """
    Compute replica-dependent step sizes.

    Parameters
    ----------
    C_stepsize_list : ndarray
        Base step-size coefficients.
    d_stepsize_list : ndarray
        Exponents controlling step-size decay.
    num_replica : int
        Number of replicas.
    param_dim : int
        Number of parameters per replica.
    beta : ndarray of shape (num_replica,) or (num_replica, 1)
        Inverse temperatures for each replica.
    num_data : int
        Number of data points.

    Returns
    -------
    step_size : ndarray of shape (num_replica, param_dim)
        Step sizes for each replica and parameter.
    """
    beta = np.asarray(beta).reshape(num_replica)
    step_size = np.zeros((num_replica, param_dim))

    for i in range(num_replica):
        effective_scale = num_data * beta[i]
        if effective_scale < 1.0:
            step_size[i, :] = C_stepsize_list
        else:
            step_size[i, :] = C_stepsize_list / (effective_scale ** d_stepsize_list)

    return step_size

