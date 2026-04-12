import numpy as np

from utilis import quad_form_via_cholesky


def build_padded_selector_vector(
    pair_mask_i,
    triplet_mask_a_i,
    triplet_mask_s_i,
    basis_switches,
    L2,
    L3,
    L2_max,
    L3_max,
):
    """
    Construct a padded binary selector vector for oscillator i.

    The block order is:
        [const]
        [2-body sin]
        [2-body cos]
        [3-body-a sin]
        [3-body-a cos]
        [3-body-s sin]
        [3-body-s cos]

    Each block is padded up to the maximum harmonic orders L2_max and L3_max.

    Parameters
    ----------
    pair_mask_i : ndarray of shape (N,)
        Binary mask for pairwise interactions affecting oscillator i.
    triplet_mask_a_i : ndarray of shape (N, N)
        Binary mask for the first type of three-body interaction.
    triplet_mask_s_i : ndarray of shape (N, N)
        Binary mask for the second type of three-body interaction.
    basis_switches : ndarray of shape (6,)
        Binary switches for enabling/disabling each basis block:
        [2-body sin, 2-body cos, 3a sin, 3a cos, 3s sin, 3s cos].
    L2, L3 : int
        Active harmonic truncation orders.
    L2_max, L3_max : int
        Maximum harmonic truncation orders.

    Returns
    -------
    selector : ndarray of shape (1 + 2*N*L2_max + 4*N*N*L3_max,)
        Padded binary selector vector.
    """
    N = triplet_mask_a_i.shape[0]
    total_len = 1 + 2 * N * L2_max + 4 * N * N * L3_max
    selector = np.zeros(total_len, dtype=float)

    selector[0] = 1.0
    pos = 1

    pair_block_len_max = N * L2_max
    pair_base = np.tile(pair_mask_i.ravel(), L2) if L2 > 0 else None

    if L2 > 0:
        selector[pos: pos + N * L2] = pair_base * float(basis_switches[0])
    pos += pair_block_len_max

    if L2 > 0:
        selector[pos: pos + N * L2] = pair_base * float(basis_switches[1])
    pos += pair_block_len_max

    triplet_block_len_max = N * N * L3_max
    triplet_base_a = np.tile(triplet_mask_a_i.T.reshape(N * N), L3) if L3 > 0 else None
    triplet_base_s = np.tile(np.triu(triplet_mask_s_i).T.reshape(N * N), L3) if L3 > 0 else None

    if L3 > 0:
        selector[pos: pos + N * N * L3] = triplet_base_a * float(basis_switches[2])
    pos += triplet_block_len_max

    if L3 > 0:
        selector[pos: pos + N * N * L3] = triplet_base_a * float(basis_switches[3])
    pos += triplet_block_len_max

    if L3 > 0:
        selector[pos: pos + N * N * L3] = triplet_base_s * float(basis_switches[4])
    pos += triplet_block_len_max

    if L3 > 0:
        selector[pos: pos + N * N * L3] = triplet_base_s * float(basis_switches[5])

    return selector


def free_energy_i(
    Y_i,
    G_i,
    pair_mask_i,
    triplet_mask_a_i,
    triplet_mask_s_i,
    tau_sq_i,
    xi_i,
    L2,
    L3,
    basis_switches,
    delta_t,
    L2_max,
    L3_max,
):
    """
    Compute the marginal free energy for a single oscillator.

    This corresponds to the negative log evidence of a Bayesian linear model
    with Gaussian noise precision xi_i and diagonal Gaussian prior variances
    tau_sq_i, after analytically integrating out the regression coefficients.
    """
    selector = build_padded_selector_vector(
        pair_mask_i,
        triplet_mask_a_i,
        triplet_mask_s_i,
        basis_switches,
        L2,
        L3,
        L2_max,
        L3_max,
    )

    active = selector == 1
    G_selected = G_i[:, active]
    tau_sq_selected = tau_sq_i[active]

    precision_prior = np.diag(1.0 / tau_sq_selected)
    gram = G_selected.T @ G_selected
    posterior_precision = gram * (delta_t ** 2) * xi_i + precision_prior

    chol = np.linalg.cholesky(posterior_precision)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))

    projected_y = G_selected.T @ Y_i
    quad_term = quad_form_via_cholesky(posterior_precision, projected_y)

    n_samples = Y_i.shape[0]
    free_energy_value = (
        0.5 * n_samples * np.log(2.0 * np.pi / xi_i)
        + 0.5 * np.sum(np.log(tau_sq_selected))
        + 0.5 * np.sum(Y_i * Y_i) * xi_i
        + 0.5 * log_det
        - 0.5 * quad_term * (delta_t ** 2) * (xi_i ** 2)
    )
    return free_energy_value


def free_energy(
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_sq,
    xi,
    L2,
    L3,
    basis_switches,
    delta_t,
    N_osci,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    """
    Compute the total free energy by summing the oscillator-wise contributions.
    """
    if (L2 < L2_min) or (L2 > L2_max) or (L3 < L3_min) or (L3 > L3_max):
        return 1e100

    total_free_energy = 0.0
    for i in range(N_osci):
        total_free_energy += free_energy_i(
            Y_i=delta_phi[:, i],
            G_i=G_L[i],
            pair_mask_i=cij[i],
            triplet_mask_a_i=cijk_a[i],
            triplet_mask_s_i=cijk_s[i],
            tau_sq_i=tau_sq[i],
            xi_i=xi[i],
            L2=L2,
            L3=L3,
            basis_switches=d_array if False else basis_switches,
            delta_t=delta_t,
            L2_max=L2_max,
            L3_max=L3_max,
        )
    return total_free_energy


def free_energy_replica(
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau_sq,
    xi,
    L2_array,
    L3_array,
    d_array,
    delta_t,
    N_osci,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    """
    Compute the free energy for all replicas.
    """
    num_replica = cij.shape[0]
    free_energy_all = np.zeros(num_replica)

    for r in range(num_replica):
        free_energy_all[r] = free_energy(
            delta_phi=delta_phi,
            G_L=G_L,
            cij=cij[r],
            cijk_a=cijk_a[r],
            cijk_s=cijk_s[r],
            tau_sq=tau_sq[r],
            xi=xi[r],
            L2=L2_array[r],
            L3=L3_array[r],
            basis_switches=d_array[r],
            delta_t=delta_t,
            N_osci=N_osci,
            L2_min=L2_min,
            L2_max=L2_max,
            L3_min=L3_min,
            L3_max=L3_max,
        )

    return free_energy_all