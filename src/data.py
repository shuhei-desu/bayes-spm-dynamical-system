import numpy as np

from get_args import get_parser


def rhs_dense(phi, omega, K2, alpha2, c_ij, K3, K3_2, alpha3, c_ijk_a, c_ijk_s):
    """
    Compute the right-hand side dphi/dt for the coupled phase dynamics.

    Parameters
    ----------
    phi : ndarray of shape (N,)
        Current phase vector.
    omega : ndarray of shape (N,)
        Natural frequencies.
    K2 : ndarray of shape (L2, N, N)
        Two-body coupling strengths.
    alpha2 : ndarray of shape (N, N) or (L2, N, N)
        Phase shifts for two-body interactions.
    c_ij : ndarray of shape (N, N)
        Binary mask for active two-body interactions.
    K3 : ndarray of shape (L3, N, N, N)
        First three-body coupling strengths.
    K3_2 : ndarray of shape (L3, N, N, N)
        Second three-body coupling strengths.
    alpha3 : ndarray of shape (N, N, N) or (L3, N, N, N)
        Phase shifts for three-body interactions.
    c_ijk_a : ndarray of shape (N, N, N)
        Binary mask for the first three-body interaction type.
    c_ijk_s : ndarray of shape (N, N, N)
        Binary mask for the second three-body interaction type.

    Returns
    -------
    ndarray of shape (N,)
        Time derivative dphi/dt.
    """
    n_oscillators = phi.shape[0]
    n_harmonics_2 = K2.shape[0]
    n_harmonics_3 = K3.shape[0]

    # ----- Two-body interaction -----
    # diff_ij[i, j] = phi_j - phi_i
    diff_ij = phi[None, None, :] - phi[None, :, None]  # (1, N, N)
    harmonics_2 = np.arange(1, n_harmonics_2 + 1)[:, None, None]  # (L2, 1, 1)
    phase_arg_2 = harmonics_2 * diff_ij  # (L2, N, N)

    alpha2_broadcast = alpha2 if alpha2.ndim == 3 else alpha2[None, :, :]
    sin_2 = np.sin(phase_arg_2 + alpha2_broadcast)
    pair_sum = (K2 * sin_2 * c_ij[None, :, :]).sum(axis=(0, 2))  # (N,)

    # ----- Three-body interaction -----
    phi_i = phi[None, :, None, None]   # (1, N, 1, 1)
    phi_j = phi[None, None, :, None]   # (1, 1, N, 1)
    phi_k = phi[None, None, None, :]   # (1, 1, 1, N)
    harmonics_3 = np.arange(1, n_harmonics_3 + 1)[:, None, None, None]  # (L3, 1, 1, 1)

    alpha3_broadcast = alpha3 if alpha3.ndim == 4 else alpha3[None, :, :, :]

    phase_arg_3a = harmonics_3 * (2 * phi_k - phi_i - phi_j) + alpha3_broadcast
    phase_arg_3s = harmonics_3 * (phi_j + phi_k - 2 * phi_i) + alpha3_broadcast

    sin_3a = np.sin(phase_arg_3a)
    sin_3s = np.sin(phase_arg_3s)

    triplet_sum_a = ((K3 * sin_3a) * c_ijk_a[None, :, :, :]).sum(axis=(0, 2, 3))
    triplet_sum_s = ((K3_2 * sin_3s) * c_ijk_s[None, :, :, :]).sum(axis=(0, 2, 3))

    return omega + pair_sum + triplet_sum_a + triplet_sum_s


def make_toy_ground_truth(
    N=4,
    L2=2,
    L3=2,
    k2_scale=1.0,
    k3_scale=1.0,
    k3_scale_2=0.5,
    alpha2ij=0.5,
    alpha3ijk=0.5,
):
    """
    Construct a fixed toy interaction structure used in the experiments.

    Notes
    -----
    This function defines a specific hand-crafted interaction graph, not a
    general random ground-truth generator.
    """
    if N < 3:
        raise ValueError("This toy ground truth requires N >= 3.")

    c_ij = np.zeros((N, N))
    c_ijk_a = np.zeros((N, N, N))
    c_ijk_s = np.zeros((N, N, N))

    c_ij[1, 0] = 1
    c_ij[2, 0] = 1
    c_ijk_a[0, 1, 2] = 1
    c_ijk_s[2, 0, 1] = 1

    K2 = k2_scale * np.ones((L2, N, N))
    K3 = k3_scale * np.ones((L3, N, N, N))
    K3_2 = k3_scale_2 * np.ones((L3, N, N, N))

    alpha2 = alpha2ij * np.ones((N, N))
    alpha3 = alpha3ijk * np.ones((N, N, N))

    K2 *= c_ij[None, :, :]
    K3 *= c_ijk_a[None, :, :, :]
    K3_2 *= c_ijk_s[None, :, :, :]

    return c_ij, c_ijk_a, c_ijk_s, K2, K3, K3_2, alpha2, alpha3


def simulate_and_make_dataset(
    N=3,
    L2=1,
    L3=1,
    T=20000,
    dt=1e-2,
    k2_scale=1.0,
    k3_scale=0.0,
    k3_scale_2=0.1,
    alpha2ij=1.0,
    alpha3ijk=1.0,
    omega_0=0.4,
    delta_omega=0.4,
    sigma_d=1.0,
    sigma_o=0.0,
    phi_ini=None,
    seed=None,
):
    """
    Simulate the phase dynamics with Euler-Maruyama integration and return
    the observed phase trajectory.
    """
    rng = np.random.default_rng(seed)

    c_ij, c_ijk_a, c_ijk_s, K2, K3, K3_2, alpha2, alpha3 = make_toy_ground_truth(
        N=N,
        L2=L2,
        L3=L3,
        k2_scale=k2_scale,
        k3_scale=k3_scale,
        k3_scale_2=k3_scale_2,
        alpha2ij=alpha2ij,
        alpha3ijk=alpha3ijk,
    )

    omega = omega_0 + delta_omega * np.arange(N)

    if phi_ini is None:
        phi_ini = np.linspace(0.0, 3.0, N)

    phi = np.array(phi_ini, dtype=float, copy=True)

    phi_traj = np.empty((T, N), dtype=float)
    time_traj = np.arange(T, dtype=float) * dt

    # Store the initial observation
    phi_traj[0] = phi + rng.normal(loc=0.0, scale=sigma_o, size=N)

    for m in range(T - 1):
        rhs = rhs_dense(phi, omega, K2, alpha2, c_ij, K3, K3_2, alpha3, c_ijk_a, c_ijk_s)
        phi = phi + dt * rhs + rng.normal(loc=0.0, scale=np.sqrt(dt) * sigma_d, size=N)
        phi_traj[m + 1] = phi + rng.normal(loc=0.0, scale=sigma_o, size=N)

    return phi_traj, time_traj


def main():
    parser = get_parser()
    args = parser.parse_args()

    phi_traj, time_traj = simulate_and_make_dataset(
        N=args.N,
        L2=args.L2,
        L3=args.L3,
        T=args.T,
        dt=args.dt,
        k2_scale=args.k2,
        k3_scale=args.k3,
        k3_scale_2=args.k3_2,
        alpha2ij=args.alpha2ij,
        alpha3ijk=args.alpha3ijk,
        omega_0=args.omega_0,
        delta_omega=args.delta_omega,
        sigma_d=args.sigma_d,
        sigma_o=args.sigma_o,
        phi_ini=np.array(args.phi_ini, dtype=float),
    )

    indices = np.linspace(0, args.T - 1, args.num_data, dtype=int)
    phis_obs = phi_traj[indices]
    time_selected = time_traj[indices]

    output_name = (
        f"data_L2={args.L2}_L3={args.L3}_k2={args.k2}_k3={args.k3}"
        f"_k3_2={args.k3_2}_alpha2ij={args.alpha2ij}_alpha3ijk={args.alpha3ijk}"
        f"_num_data={args.num_data}_sigma_d={args.sigma_d}_sigma_o={args.sigma_o}.npz"
    )

    np.savez(
        output_name,
        time_selected=time_selected,
        phis_obs=phis_obs,
    )


if __name__ == "__main__":
    main()

