import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Phi oscillator simulation")
    ##setting of data
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--omega_0", type=float, default=0.4)
    parser.add_argument("--delta_omega", type=float, default=0.4)
    parser.add_argument("--L2", type=int, default=1)
    parser.add_argument("--L3", type=int, default=1)
    parser.add_argument("--k2", type=float, default=0.5)
    parser.add_argument("--k3", type=float, default=0.5)
    parser.add_argument("--k3_2", type=float, default=0.5)
    parser.add_argument("--alpha2ij", type=float, default=0.0)
    parser.add_argument("--alpha3ijk", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=20000)
    parser.add_argument("--num_data", type = int, default=2000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma_d", type=float, default=0.1)
    parser.add_argument("--sigma_o", type=float, default=0.0)
    parser.add_argument("--phi_ini", nargs="+", type=float, default=[0.0, 2.0, 4.0])

    ##setting of emc
    parser.add_argument("--iterate", type=int, default=100000)
    parser.add_argument("--burn_in", type=int, default=50000)
    parser.add_argument("--p_star", type=float, default=0.3)
    parser.add_argument("--tweak", type=float, default=2.0)
    parser.add_argument("--L2_max", type=int, default=3)
    parser.add_argument("--L3_max", type=int, default=3)
    parser.add_argument("--L2_min", type=int, default=1)
    parser.add_argument("--L3_min", type=int, default=1)



    return parser