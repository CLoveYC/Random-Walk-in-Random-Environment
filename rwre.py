from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "rwre_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
SEED = 42
T = 2000               # trajectory length
L = 200                # potential plotted on [-L, L]
RWRE_MODE = "sinai"    # "sinai" or "transient"
SINAI_P_VALUES = (0.2, 0.8)
SCALING_T_VALUES = np.array([100, 300, 1000, 3000, 10000, 30000, 100000])
SCALING_TRIALS = 100

rng_env = np.random.default_rng(SEED)
rng_simple_walk = np.random.default_rng(SEED + 1)
rng_rwre_walk = np.random.default_rng(SEED + 2)
rng_scaling = np.random.default_rng(SEED + 3)


# ============================================================
# Environment sampling
# ============================================================
def sample_p(rng, mode="sinai"):
    """
    Sample a right-step probability p_x for one site x.

    mode = "sinai":
        p_x = 0.2 or 0.8 with equal probability
        => E[log rho] = 0  (recurrent Sinai-type RWRE)

    mode = "transient":
        p_x = 0.9 with prob 0.2, 0.4 with prob 0.8
        => E[p] = 0.5 but E[log rho] < 0  (right transient)
    """
    if mode == "sinai":
        return SINAI_P_VALUES[0] if rng.random() < 0.5 else SINAI_P_VALUES[1]
    elif mode == "transient":
        return 0.9 if rng.random() < 0.2 else 0.4
    else:
        raise ValueError("mode must be 'sinai' or 'transient'")


def build_environment_on_interval(L, rng, mode="sinai"):
    """
    Build an environment {p_x}_{x=-L,...,L}.
    """
    env = {}
    for x in range(-L, L + 1):
        env[x] = sample_p(rng, mode=mode)
    return env


def get_or_sample_p(env, x, rng, mode="sinai"):
    """
    Return the frozen p_x at site x, sampling it once if needed.
    """
    if x not in env:
        env[x] = sample_p(rng, mode=mode)
    return env[x]


# ============================================================
# Potential
# ============================================================
def rho(p):
    return (1 - p) / p


def compute_potential(env, L):
    """
    Compute V(x) on x in [-L, L] with V(0)=0.

    For x > 0:
        V(x) = sum_{i=1}^x log rho_i

    For x < 0:
        V(x) = - sum_{i=x+1}^0 log rho_i

    This gives a convenient two-sided potential with V(0)=0.
    """
    xs = np.arange(-L, L + 1)
    V = np.zeros(len(xs))
    idx0 = L  # position of x = 0 in xs

    # positive side
    s = 0.0
    for x in range(1, L + 1):
        s += np.log(rho(env[x]))
        V[idx0 + x] = s

    # negative side
    s = 0.0
    for x in range(-1, -L - 1, -1):
        s -= np.log(rho(env[x + 1]))
        V[idx0 + x] = s

    return xs, V


# ============================================================
# Trajectory simulation
# ============================================================
def simulate_simple_rw(T, rng):
    """
    Simple symmetric random walk.
    """
    X = np.zeros(T + 1, dtype=int)
    for t in range(T):
        step = rng.choice([-1, 1])
        X[t + 1] = X[t] + step
    return X


def simulate_rwre(T, rng_walk, env, rng_env, mode="sinai"):
    """
    Simulate 1D RWRE.
    The supplied environment is frozen and shared with the potential plot.
    If the walk exits the plotted interval, new sites are sampled once and
    then kept fixed.
    """
    X = np.zeros(T + 1, dtype=int)

    for t in range(T):
        x = X[t]
        p = get_or_sample_p(env, x, rng_env, mode=mode)
        step = 1 if rng_walk.random() < p else -1
        X[t + 1] = x + step

    return X


def simulate_simple_rw_at_times(t_values, rng):
    """
    Simulate one ordinary walk and record X_t at selected increasing times.
    """
    targets = np.asarray(t_values)
    steps = rng.choice(np.array([-1, 1]), size=int(targets[-1]))
    path = np.cumsum(steps)
    return path[targets - 1]


def simulate_rwre_at_times(t_values, rng_walk, rng_env, mode="sinai"):
    """
    Simulate one RWRE in one frozen environment and record X_t at selected times.
    """
    targets = np.asarray(t_values)
    env = {}
    X = 0
    out = np.zeros(len(targets), dtype=int)
    target_idx = 0

    for t in range(1, int(targets[-1]) + 1):
        p = get_or_sample_p(env, X, rng_env, mode=mode)
        X += 1 if rng_walk.random() < p else -1
        if t == targets[target_idx]:
            out[target_idx] = X
            target_idx += 1
            if target_idx == len(targets):
                break

    return out


def run_scaling_experiment(t_values, trials, rng, mode="sinai"):
    """
    Estimate typical displacement by the median of |X_T| over many runs.
    Each RWRE run gets its own frozen environment.
    """
    simple_abs = np.zeros((trials, len(t_values)), dtype=float)
    rwre_abs = np.zeros((trials, len(t_values)), dtype=float)

    for trial in range(trials):
        rng_simple = np.random.default_rng(rng.integers(0, 2**63))
        rng_walk = np.random.default_rng(rng.integers(0, 2**63))
        rng_environment = np.random.default_rng(rng.integers(0, 2**63))

        simple_positions = simulate_simple_rw_at_times(t_values, rng_simple)
        rwre_positions = simulate_rwre_at_times(t_values, rng_walk, rng_environment, mode=mode)
        simple_abs[trial] = np.abs(simple_positions)
        rwre_abs[trial] = np.abs(rwre_positions)

    return np.median(simple_abs, axis=0), np.median(rwre_abs, axis=0)


def scaled_reference(values, reference_shape):
    """
    Scale a reference curve so its last point matches the measured last point.
    """
    scale = values[-1] / reference_shape[-1]
    return scale * reference_shape


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Use one fixed environment for both the potential plot and the RWRE path.
    env = build_environment_on_interval(L, rng_env, mode=RWRE_MODE)
    xs, V_rwre = compute_potential(env, L)
    V_rw = np.zeros_like(xs, dtype=float)

    # For trajectory plot
    X_rw = simulate_simple_rw(T, rng_simple_walk)
    X_rwre = simulate_rwre(T, rng_rwre_walk, env, rng_env, mode=RWRE_MODE)

    ps = np.array([env[x] for x in range(-L, L + 1)])
    log_rhos = np.log((1 - ps) / ps)
    print(f"mode = {RWRE_MODE}")
    print(f"plotted environment: mean p = {ps.mean():.4f}, mean log rho = {log_rhos.mean():.4f}")
    print(f"RWRE path range: [{X_rwre.min()}, {X_rwre.max()}]")

    # --------------------------------------------------------
    # Figure 1: Potential comparison
    # --------------------------------------------------------
    plt.figure(figsize=(9, 4.8))
    plt.plot(xs, V_rw, label="Ordinary RW potential", linewidth=2)
    plt.plot(xs, V_rwre, label="RWRE potential", linewidth=2)
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("position x")
    plt.ylabel("potential V(x)")
    plt.title("Potential: Ordinary RW vs RWRE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure_1.png", dpi=160)
    plt.close()

    # --------------------------------------------------------
    # Figure 2: Trajectory comparison
    # --------------------------------------------------------
    ts = np.arange(T + 1)

    plt.figure(figsize=(9, 4.8))
    plt.plot(ts, X_rw, label="Ordinary RW path", linewidth=1.8)
    plt.plot(ts, X_rwre, label="RWRE path", linewidth=1.8)
    plt.xlabel("time t")
    plt.ylabel("position X_t")
    plt.title("Trajectory: Ordinary RW vs RWRE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure_2.png", dpi=160)
    plt.close()

    # --------------------------------------------------------
    # Figure 3: Scaling comparison
    # --------------------------------------------------------
    median_simple, median_rwre = run_scaling_experiment(
        SCALING_T_VALUES,
        SCALING_TRIALS,
        rng_scaling,
        mode=RWRE_MODE,
    )

    sqrt_ref = scaled_reference(median_simple, np.sqrt(SCALING_T_VALUES))
    log2_ref = scaled_reference(median_rwre, np.log(SCALING_T_VALUES) ** 2)

    print("\nScaling experiment: median |X_T|")
    print("T ordinary_RW RWRE")
    for t, ordinary, rwre in zip(SCALING_T_VALUES, median_simple, median_rwre):
        print(f"{t} {ordinary:.1f} {rwre:.1f}")

    plt.figure(figsize=(9, 4.8))
    plt.loglog(
        SCALING_T_VALUES,
        median_simple,
        marker="o",
        linewidth=2,
        label="ordinary RW median |X_T|",
    )
    plt.loglog(
        SCALING_T_VALUES,
        median_rwre,
        marker="o",
        linewidth=2,
        label="Sinai RWRE median |X_T|",
    )
    plt.loglog(
        SCALING_T_VALUES,
        sqrt_ref,
        linestyle="--",
        linewidth=1.6,
        label=r"reference $\sqrt{T}$",
    )
    plt.loglog(
        SCALING_T_VALUES,
        log2_ref,
        linestyle="--",
        linewidth=1.6,
        label=r"reference $(\log T)^2$",
    )
    plt.xlabel("time T")
    plt.ylabel(r"median displacement $|X_T|$")
    plt.title("Scaling: Ordinary RW vs Sinai RWRE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure_3_scaling.png", dpi=160)
    plt.close()
