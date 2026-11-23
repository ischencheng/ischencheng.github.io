"""Numeric checks of the formulas in the post. numpy only.

    uv run --with numpy python check_identities.py

A  one grid recurrence covers both a diffusion-forcing-style sampler (token x pass)
   and a looped AttnRes network (layer x loop); with tied cell weights, a
   predecessor-only same-pass read and a self-only previous-pass read, the two
   are the same function.
B  an execution table S[k, i] reproduces a prelude / core x 3 / coda forward.
C  parallel passes of a latent-feedback model are Jacobi sweeps of a triangular
   system: after p passes positions t <= p equal token-by-token decoding.
D  the depth identities of section 1: a separable depth kernel folds into a
   fixed-size recurrence, and zero-initialised AttnRes is the PreNorm residual.

Prints the largest error of each check and ALL CHECKS PASS, or exits non-zero.
"""
import sys

import numpy as np

TOL = 1e-10
rng = np.random.default_rng(0)
d = 8
failures = []


def report(name, err, tol=TOL, want_small=True):
    ok = (err < tol) if want_small else (err > tol)
    print(f"{name:66s} {err:.3e}  {'ok' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def rms_norm(x, eps=1e-12):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# The template.  u[i] is the state of cell i (None = never computed).
#   u[i, k] = F_i( read_same(i, cells j < i as they stand in pass k),
#                  read_prev(i, all cells as they stood after pass k - 1),
#                  c(i, k) )
# S[k][i] says whether cell i is updated in pass k; a cell that is not updated
# keeps its value.
# ---------------------------------------------------------------------------
def run_grid(n_cells, n_pass, F, read_same, read_prev, cond, S, u0):
    u = list(u0)
    history = []
    for k in range(n_pass):
        prev = list(u)
        for i in range(n_cells):
            if S[k][i]:
                u[i] = F(i, read_same(i, u[:i]), read_prev(i, prev), cond(i, k))
        history.append(list(u))
    return history


# ---------------------------------------------------------------------------
# A1. Diffusion-forcing-style sampler, written the usual way: rows are passes,
#     inside a row go left to right; an RNN latent z carries the earlier tokens.
#       z_t = f(z_{t-1}, x_t, k)      x_t <- g(z_t)
# ---------------------------------------------------------------------------
T, M = 5, 4
Wz, Wx, Wg = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
bk = rng.normal(size=d)
x_init = rng.normal(size=(T, d))  # pure noise


def df_sampler():
    x = x_init.copy()
    zs = []
    for m in range(M):
        z = np.zeros(d)
        row = []
        for t in range(T):
            z = np.tanh(Wz @ z + Wx @ x[t] + bk * (M - m))
            x[t] = Wg @ z
            row.append(z.copy())
        zs.append(row)
    return np.array(zs)


def cell_df(i, same, prev, c):
    return np.tanh(Wz @ same + Wx @ prev + bk * c)


direct = df_sampler()
# cell state u = z; the cell's own previous value is x = g(z) (x_init before pass 1).
tmpl = run_grid(
    T, M, cell_df,
    read_same=lambda i, cur: cur[i - 1] if i > 0 else np.zeros(d),
    read_prev=lambda i, prev: x_init[i] if prev[i] is None else Wg @ prev[i],
    cond=lambda i, k: M - k,
    S=[[1] * T] * M, u0=[None] * T)
report("A1 template == diffusion-forcing sampler (token x pass)", np.abs(np.array(tmpl) - direct).max())

# ---------------------------------------------------------------------------
# A2. Looped AttnRes network, written the usual way: for each loop, for each
#     layer: depth softmax over the embedding and this loop's earlier layers,
#     plus a depth softmax over the previous loop's layers, plus a loop index.
# ---------------------------------------------------------------------------
L, R = 5, 3
emb = rng.normal(size=d)
Wl = rng.normal(size=(L, d, d)) / np.sqrt(d)   # one weight per layer
Vl = rng.normal(size=(L, d, d)) / np.sqrt(d)
q_same = rng.normal(size=(L, d))                # static pseudo-queries
q_prev = rng.normal(size=(L, d))
loop_emb = rng.normal(size=(R, d))


def depth_read(q, srcs):
    srcs = np.stack(srcs)
    return softmax(rms_norm(srcs) @ q) @ srcs


def loop_attnres():
    prev = None
    out = []
    for r in range(R):
        cur = []
        for l in range(L):
            x = depth_read(q_same[l], [emb] + cur)
            m = np.zeros(d) if prev is None else depth_read(q_prev[l], prev)
            cur.append(np.tanh(Wl[l] @ x + Vl[l] @ m + loop_emb[r]))
        prev = cur
        out.append(list(cur))
    return np.array(out)


direct = loop_attnres()
tmpl = run_grid(
    L, R,
    F=lambda i, same, prev, c: np.tanh(Wl[i] @ same + Vl[i] @ prev + c),
    read_same=lambda i, cur: depth_read(q_same[i], [emb] + cur),
    read_prev=lambda i, prev: np.zeros(d) if prev[0] is None else depth_read(q_prev[i], prev),
    cond=lambda i, k: loop_emb[k],
    S=[[1] * L] * R, u0=[None] * L)
report("A2 template == looped AttnRes network (layer x loop)", np.abs(np.array(tmpl) - direct).max())

# ---------------------------------------------------------------------------
# A3. Remove the three differences (cell weights tied along i, same-pass read =
#     predecessor only, previous-pass read = own value only) and the looped
#     network is the sampler of A1 with layers in the place of tokens.
# ---------------------------------------------------------------------------
def loop_net_reduced():
    v_prev = [None] * T
    out = []
    for r in range(M):
        cur = []
        for l in range(T):
            below = cur[l - 1] if l > 0 else np.zeros(d)
            own = x_init[l] if v_prev[l] is None else Wg @ v_prev[l]
            cur.append(np.tanh(Wz @ below + Wx @ own + bk * (M - r)))
        v_prev = cur
        out.append(list(cur))
    return np.array(out)


report("A3 reduced looped network == diffusion-forcing sampler", np.abs(loop_net_reduced() - df_sampler()).max())

# ---------------------------------------------------------------------------
# B. Execution table.  4 blocks: prelude (once), two core blocks (3 loops),
#    coda (once, after the last loop).  Direct forward vs the template with S.
# ---------------------------------------------------------------------------
NB, NL = 4, 3
Wb = rng.normal(size=(NB, d, d)) / np.sqrt(d)
Vb = rng.normal(size=(NB, d, d)) / np.sqrt(d)
qs = rng.normal(size=(NB, d))
qp = rng.normal(size=(NB, d))


def block(i, same_srcs, prev_core):
    x = depth_read(qs[i], same_srcs)
    m = np.zeros(d) if prev_core is None else depth_read(qp[i], prev_core)
    return np.tanh(Wb[i] @ x + Vb[i] @ m)


def prelude_core_coda():
    b0 = block(0, [emb], None)
    prev = None
    for _ in range(NL):
        b1 = block(1, [emb, b0], prev)
        b2 = block(2, [emb, b0, b1], prev)
        prev = [b1, b2]
    b3 = block(3, [emb, b0, b1, b2], None)
    return np.stack([b0, b1, b2, b3])


S_table = [[1, 1, 1, 0],
           [0, 1, 1, 0],
           [0, 1, 1, 1]]
hist = run_grid(
    NB, NL,
    F=lambda i, same, prev, c: np.tanh(Wb[i] @ same + Vb[i] @ prev),
    read_same=lambda i, cur: depth_read(qs[i], [emb] + cur),
    read_prev=lambda i, prev: (np.zeros(d) if i in (0, 3) or prev[1] is None
                               else depth_read(qp[i], [prev[1], prev[2]])),
    cond=lambda i, k: None,
    S=S_table, u0=[None] * NB)
report("B  execution table == prelude / core x 3 / coda forward",
       np.abs(np.stack(hist[-1]) - prelude_core_coda()).max())

# ---------------------------------------------------------------------------
# C. Latent feedback.  Position t reads e_t + W h_{t-1}; the stack G is causal
#    (two layers of causal attention + MLP), so h_t depends on inputs 1..t.
# ---------------------------------------------------------------------------
Tn = 8
E = rng.normal(size=(Tn, d))
Wfb = rng.normal(size=(d, d)) / np.sqrt(d)
layers = [tuple(rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)) for _ in range(2)]


def causal_stack(u):
    """Rows of u are positions; row t of the result depends on rows <= t only."""
    h = u.copy()
    n = len(h)
    for Wq, Wk, Wv, Wm in layers:
        out = np.zeros_like(h)
        for t in range(n):
            a = softmax((h[: t + 1] @ Wk.T) @ (Wq @ h[t]) / np.sqrt(d))
            out[t] = h[t] + a @ (h[: t + 1] @ Wv.T)
        h = out + np.tanh(out @ Wm.T)
    return h


def decode_sequential():
    h_prev = np.zeros(d)
    u = np.zeros((0, d))
    hs = []
    for t in range(Tn):
        u = np.vstack([u, E[t] + Wfb @ h_prev])
        h_prev = causal_stack(u)[t]
        hs.append(h_prev)
    return np.array(hs)


exact = decode_sequential()
h = np.zeros((Tn, d))
jacobi_err = []
worst_done, least_not_done = 0.0, np.inf
for p in range(1, Tn + 1):
    shifted = np.vstack([np.zeros((1, d)), h[:-1]])     # h_{t-1} from the previous pass
    h = causal_stack(E + shifted @ Wfb.T)
    err = np.abs(h - exact).max(-1)
    jacobi_err.append(err)
    worst_done = max(worst_done, err[:p].max())
    if p < Tn:
        least_not_done = min(least_not_done, err[p:].max())
report("C  after p parallel passes, positions t <= p are exact", worst_done)
report("C  while some position t > p still differs (largest error per pass)", least_not_done, 1e-6, want_small=False)
np.save("jacobi_err.npy", np.array(jacobi_err))

# ---------------------------------------------------------------------------
# D. Section 1 of the post (identities from the AttnRes paper, arXiv 2603.15031).
#    D1  a separable depth kernel folds into a fixed-size recurrence;
#    D2  a zero-initialised pseudo-query makes AttnRes equal to the PreNorm
#        residual once the sublayer's RMSNorm is applied.
# ---------------------------------------------------------------------------
Ld = 12
v = rng.normal(size=(Ld, d)) * rng.uniform(0.5, 4.0, size=(Ld, 1))   # v[0] = embedding
feat = lambda x: np.tanh(x[:5])
qd = rng.normal(size=(Ld + 1, d))
state = np.zeros((5, d))
err = 0.0
for l in range(1, Ld + 1):
    state = state + np.outer(feat(v[l - 1]), v[l - 1])               # write
    direct = sum((feat(qd[l]) @ feat(v[i])) * v[i] for i in range(l))
    err = max(err, np.abs(feat(qd[l]) @ state - direct).max())       # read
report("D1 separable depth kernel == fixed-size recurrence", err)
uniform = softmax(rms_norm(v, 1e-12) @ np.zeros(d)) @ v              # zero query: uniform mean
report("D2 zero-init AttnRes == PreNorm residual after RMSNorm",
       np.abs(rms_norm(uniform) - rms_norm(v.sum(0))).max())

if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("ALL CHECKS PASS")
