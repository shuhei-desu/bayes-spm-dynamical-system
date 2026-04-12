import time
import numpy as np

from utilis import (
    build_feature_matrix_per_oscillator,
    compute_phase_increments,
    precompute_symmetric_pairs,
    compute_step_sizes,
)
from free_energy import free_energy_replica
from mcmc import ms_cij, ms_cijk_a, ms_cijk_s, ms_tau, ms_xi, ms_L2, ms_L3, ms_d
from get_args import get_parser


# =========================================================
# File names
# =========================================================

def build_data_filename(args):
    return (
        f"data_L2={args.L2}_L3={args.L3}_k2={args.k2}_k3={args.k3}"
        f"_k3_2={args.k3_2}_alpha2ij={args.alpha2ij}_alpha3ijk={args.alpha3ijk}"
        f"_num_data={args.num_data}_sigma_d={args.sigma_d}_sigma_o={args.sigma_o}.npz"
    )


def build_emc_filename(args):
    return (
        f"emc_L2={args.L2}_L3={args.L3}_k2={args.k2}_k3={args.k3}"
        f"_k3_2={args.k3_2}_alpha2ij={args.alpha2ij}_alpha3ijk={args.alpha3ijk}"
        f"_num_data={args.num_data}_sigma_d={args.sigma_d}_sigma_o={args.sigma_o}.npz"
    )


# =========================================================
# Initialization
# =========================================================

def initialize_beta(num_replica):
    beta = np.array([1.3 ** (i - num_replica + 1) for i in range(num_replica)], dtype=float)
    beta[0] = 0.0
    return beta.reshape(-1, 1)


def initialize_binary_structures(num_replica, n_osci):
    cij = np.random.randint(0, 2, (num_replica, n_osci, n_osci)).astype(float)
    cijk_a = np.random.randint(0, 2, (num_replica, n_osci, n_osci, n_osci)).astype(float)
    cijk_s = np.random.randint(0, 2, (num_replica, n_osci, n_osci, n_osci)).astype(float)

    for r in range(num_replica):
        np.fill_diagonal(cij[r], 0.0)

    i, j, k = np.indices((n_osci, n_osci, n_osci))
    mask = (i != j) & (j != k) & (i != k)
    mask_s = mask & (j < k)

    cijk_a *= mask[None, :, :, :]
    cijk_s *= mask_s[None, :, :, :]

    return cij, cijk_a, cijk_s


def compute_free_energy_all(
    delta_phi,
    G_L,
    cij,
    cijk_a,
    cijk_s,
    tau,
    log_xi,
    L2,
    L3,
    d,
    delta_t,
    n_osci,
    L2_min,
    L2_max,
    L3_min,
    L3_max,
):
    return free_energy_replica(
        delta_phi,
        G_L,
        cij,
        cijk_a,
        cijk_s,
        tau * tau,
        np.exp(log_xi),
        L2,
        L3,
        d,
        delta_t,
        n_osci,
        L2_min,
        L2_max,
        L3_min,
        L3_max,
    )


# =========================================================
# Main
# =========================================================

def main():
    args = get_parser().parse_args()

    # ---------- load ----------
    data = np.load(build_data_filename(args))
    phi_traj = data["phis_obs"]
    time_traj = data["time_selected"]

    # ---------- preprocess ----------
    G_L = build_feature_matrix_per_oscillator(phi_traj, args.L2_max, args.L3_max)
    delta_phi, delta_t = compute_phase_increments(phi_traj, time_traj)

    n_osci = phi_traj.shape[1]
    n_data = len(delta_phi)
    num_replica = 40

    tau_len = 1 + 2 * n_osci * args.L2_max + 4 * n_osci * n_osci * args.L3_max
    d_len = 6

    beta = initialize_beta(num_replica)
    J_s, K_s = precompute_symmetric_pairs(n_osci)

    # ---------- step size ----------
    step_width_tau = compute_step_sizes(
        np.ones(tau_len),
        np.zeros(tau_len),
        num_replica,
        tau_len,
        beta,
        n_data,
    )
    step_width_tau = np.broadcast_to(
        step_width_tau[:, None, :].copy(),
        (num_replica, n_osci, tau_len),
    )

    step_width_xi = compute_step_sizes(
        np.full(n_osci, 5.0),
        np.full(n_osci, 0.6),
        num_replica,
        n_osci,
        beta,
        n_data,
    )

    # ---------- init ----------
    cij = np.random.randint(0, 2, (num_replica, n_osci, n_osci)).astype(float)
    cijk_a = np.random.randint(0, 2, (num_replica, n_osci, n_osci, n_osci)).astype(float)
    cijk_s = np.random.randint(0, 2, (num_replica, n_osci, n_osci, n_osci)).astype(float)
    cij, cijk_a, cijk_s = initialize_binary_structures(num_replica, n_osci)

    tau = np.random.uniform(0.01, 10.0, (num_replica, n_osci, tau_len))
    log_xi = np.log(np.random.uniform(0.03, 1500.0, (num_replica, n_osci)))
    L2 = np.random.randint(args.L2_min, args.L2_max + 1, num_replica)
    L3 = np.random.randint(args.L3_min, args.L3_max + 1, num_replica)
    d = np.random.randint(0, 2, (num_replica, d_len)).astype(float)

    F = compute_free_energy_all(
        delta_phi,
        G_L,
        cij,
        cijk_a,
        cijk_s,
        tau,
        log_xi,
        L2,
        L3,
        d,
        delta_t,
        n_osci,
        args.L2_min,
        args.L2_max,
        args.L3_min,
        args.L3_max,
    )

    # ---------- history ----------
    iterate = args.iterate
    burn_in = args.burn_in
    p_star = args.p_star
    tweak = args.tweak

    all_F_emc = np.zeros((iterate, 1))
    all_cij_emc = np.zeros((iterate, n_osci, n_osci))
    all_cijk_a_emc = np.zeros((iterate, n_osci, n_osci, n_osci))
    all_cijk_s_emc = np.zeros((iterate, n_osci, n_osci, n_osci))
    all_tau_emc = np.zeros((iterate, n_osci, tau_len))
    all_log_xi_emc = np.zeros((iterate, n_osci))
    all_L2_emc = np.zeros(iterate)
    all_L3_emc = np.zeros(iterate)
    all_d_emc = np.zeros((iterate, d_len))

    cij_count = np.zeros((num_replica, n_osci, n_osci))
    cij_accept = np.zeros((num_replica, n_osci, n_osci))
    cijk_a_count = np.zeros((num_replica, n_osci, n_osci, n_osci))
    cijk_a_accept = np.zeros((num_replica, n_osci, n_osci, n_osci))
    cijk_s_count = np.zeros((num_replica, n_osci, n_osci, n_osci))
    cijk_s_accept = np.zeros((num_replica, n_osci, n_osci, n_osci))
    tau_count = np.zeros((num_replica, n_osci, tau_len))
    tau_accept = np.zeros((num_replica, n_osci, tau_len))
    xi_count = np.zeros((num_replica, n_osci))
    xi_accept = np.zeros((num_replica, n_osci))
    L2_count = np.zeros(num_replica)
    L2_accept = np.zeros(num_replica)
    L3_count = np.zeros(num_replica)
    L3_accept = np.zeros(num_replica)
    d_count = np.zeros((num_replica, d_len))
    d_accept = np.zeros((num_replica, d_len))
    num_ex = np.zeros(num_replica - 1)
    accept_num_ex = np.zeros(num_replica - 1)

    # 初期値保存
    all_F_emc[0, :] = np.asarray(F[-1]).reshape(1)
    all_cij_emc[0] = cij[-1]
    all_cijk_a_emc[0] = cijk_a[-1]
    all_cijk_s_emc[0] = cijk_s[-1]
    all_tau_emc[0] = tau[-1]
    all_log_xi_emc[0] = log_xi[-1]
    all_L2_emc[0] = L2[-1]
    all_L3_emc[0] = L3[-1]
    all_d_emc[0] = d[-1]

    # ---------- update selection ----------
    weights = np.array([
        n_osci * (n_osci - 1),
        n_osci * (n_osci - 1) * (n_osci - 2),
        n_osci * (n_osci - 1) * (n_osci - 2) / 2,
        tau_len * n_osci,
        n_osci,
        1,
        1,
        d_len,
    ], dtype=float)
    probs = weights / weights.sum()
    options = ["cij", "cijk_a", "cijk_s", "tau", "xi", "L2", "L3", "d"]

    # =========================================================
    # MCMC loop
    # =========================================================

    t0 = time.time()

    for t in range(1, iterate):
        selected = np.random.choice(options, p=probs)

        if selected == "cij":
            cij_next, F_next_step, proposed, accepted = ms_cij(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max
            )
            cij_count += proposed
            cij_accept += accepted
            cij = cij_next.copy()

        elif selected == "cijk_a":
            cijk_a_next, F_next_step, proposed, accepted = ms_cijk_a(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max
            )
            cijk_a_count += proposed
            cijk_a_accept += accepted
            cijk_a = cijk_a_next.copy()

        elif selected == "cijk_s":
            cijk_s_next, F_next_step, proposed, accepted = ms_cijk_s(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max,
                J_s, K_s
            )
            cijk_s_count += proposed
            cijk_s_accept += accepted
            cijk_s = cijk_s_next.copy()

        elif selected == "tau":
            tau_next, F_next_step, proposed, accepted = ms_tau(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max
            )
            tau_count += proposed
            tau_accept += accepted
            tau = tau_next.copy()

        elif selected == "xi":
            log_xi_next, F_next_step, proposed, accepted = ms_xi(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max
            )
            xi_count += proposed
            xi_accept += accepted
            log_xi = log_xi_next.copy()

        elif selected == "L2":
            L2_next, F_next_step, replica_new = ms_L2(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max
            )
            L2_count += np.ones(num_replica)
            L2_accept += replica_new
            L2 = L2_next.copy()

        elif selected == "L3":
            L3_next, F_next_step, replica_new = ms_L3(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max
            )
            L3_count += np.ones(num_replica)
            L3_accept += replica_new
            L3 = L3_next.copy()

        elif selected == "d":
            d_next, F_next_step, proposed, accepted = ms_d(
                F, num_replica, delta_phi, G_L,
                cij, cijk_a, cijk_s, tau, log_xi,
                L2, L3, d,
                step_width_tau, step_width_xi,
                delta_t, n_data, n_osci,
                beta, args.L2_min, args.L2_max,
                args.L3_min, args.L3_max
            )
            d_count += proposed
            d_accept += accepted
            d = d_next.copy()

        if t % np.sum(weights).astype(int) == 0:
            for l in range(num_replica - 1):
                v = (beta[l + 1] - beta[l]) * (F_next_step[l + 1] - F_next_step[l])
                num_ex[l] += 1.0
                if v > np.log(np.random.uniform(0, 1)):
                    cij[[l, l + 1]] = cij[[l + 1, l]]
                    cijk_a[[l, l + 1]] = cijk_a[[l + 1, l]]
                    cijk_s[[l, l + 1]] = cijk_s[[l + 1, l]]
                    tau[[l, l + 1]] = tau[[l + 1, l]]
                    log_xi[[l, l + 1]] = log_xi[[l + 1, l]]
                    L2[[l, l + 1]] = L2[[l + 1, l]]
                    L3[[l, l + 1]] = L3[[l + 1, l]]
                    d[[l, l + 1]] = d[[l + 1, l]]
                    F_next_step[[l, l + 1]] = F_next_step[[l + 1, l]]
                    accept_num_ex[l] += 1.0

        F = F_next_step.copy()

        if (t % (10 * np.sum(weights).astype(int)) == 0) and (t != 0) and (t < burn_in):
            log_step_width_tau = np.log(step_width_tau) + 5.0 * (
                tau_accept / np.where(tau_count == 0, 1, tau_count) - p_star
            ) / (t + tweak)
            step_width_tau = np.exp(log_step_width_tau)

            log_step_width_xi = np.log(step_width_xi) + 5.0 * (
                xi_accept / np.where(xi_count == 0, 1, xi_count) - p_star
            ) / (t + tweak)
            step_width_xi = np.exp(log_step_width_xi)

        all_F_emc[t, :] = np.asarray(F[-1]).reshape(1)
        all_cij_emc[t] = cij[-1]
        all_cijk_a_emc[t] = cijk_a[-1]
        all_cijk_s_emc[t] = cijk_s[-1]
        all_tau_emc[t] = tau[-1]
        all_log_xi_emc[t] = log_xi[-1]
        all_L2_emc[t] = L2[-1]
        all_L3_emc[t] = L3[-1]
        all_d_emc[t] = d[-1]

        if t % 100 == 0:
            print(f"step={t}, F={float(F[-1]):.6f}")

    print(f"Done. {time.time() - t0:.2f} sec")

    meta5_names = np.array(["delta_t", "L2_max", "L2_min", "L3_max", "L3_min"])
    meta5 = np.array(
        [
            float(delta_t),
            float(args.L2_max),
            float(args.L2_min),
            float(args.L3_max),
            float(args.L3_min),
        ],
        dtype=np.float64,
    )

    np.savez_compressed(
        build_emc_filename(args),
        F=all_F_emc,
        cij=all_cij_emc,
        cijk_a=all_cijk_a_emc,
        cijk_s=all_cijk_s_emc,
        tau=all_tau_emc,
        log_xi=all_log_xi_emc,
        L2=all_L2_emc,
        L3=all_L3_emc,
        d=all_d_emc,
        meta5=meta5,
        meta5_names=meta5_names,
    )


if __name__ == "__main__":
    main()