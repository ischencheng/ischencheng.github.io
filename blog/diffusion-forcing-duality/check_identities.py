"""Numeric checks of the formulas in the post. numpy only.

    uv run --with numpy python check_identities.py

A  one grid recurrence covers both diffusion forcing sampling (token x pass,
   written as Algorithm 2 of arXiv 2407.01392) and a looped AttnRes network
   (layer x loop); with tied cell weights, a predecessor-only same-pass read and
   a self-only previous-pass read, the two are the same function.
B  an execution table S[k, i] reproduces a prelude / core x 3 / coda forward.
C  parallel passes of a latent-feedback model are Jacobi sweeps of a triangular
   system: after p passes positions t <= p equal token-by-token decoding.
D  the depth identities of section 1: a separable depth kernel folds into a
   fixed-size recurrence, and zero-initialised AttnRes is the PreNorm residual.
E  coarse to fine.  A looped block-AttnRes network written at the granularity of
   real implementations (sublayers, block sums, a restart from the prelude state,
   messages through a keep/gain gate, or one softmax over both passes' sources)
   is the same grid recurrence with one cell per sublayer; one arrow of the
   coarse picture is one weight shared by every sublayer of the source block.

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
def run_grid(n_cells, n_pass, F, read_same, read_prev, cond, S, u0, cond_in_reads=False):
    u = list(u0)
    history = []
    for k in range(n_pass):
        prev = list(u)
        for i in range(n_cells):
            if S[k][i]:
                c = cond(i, k)
                same = read_same(i, u[:i], c) if cond_in_reads else read_same(i, u[:i])
                before = read_prev(i, prev, c) if cond_in_reads else read_prev(i, prev)
                u[i] = F(i, same, before, c)
        history.append(list(u))
    return history


# ---------------------------------------------------------------------------
# A1. Diffusion forcing sampling as the paper's Algorithm 2, with toy functions
#     in the place of the transition p_theta and the noise prediction eps_theta
#     and the noise w drawn in advance.  K is the scheduling matrix; here a
#     pyramid, so the tokens of one row sit at different noise levels.
#       for row m = M-1 .. 0:   for t = 1 .. T:
#           z_t <- p(z_{t-1}, x_t, K[m+1, t])
#           k   <- K[m, t]
#           x_t <- a_k (x_t - b_k eps(z_t, x_t, k)) + s_k w
#     A cell keeps two values: z_t, read by the next token of the same row, and
#     x_t, read by the same token in the next row.  z_t of the previous row is
#     overwritten without being read.
# ---------------------------------------------------------------------------
T, LEVELS = 5, 3
M = LEVELS + T - 1
K_sched = np.array([[min(LEVELS, max(0, m - (T - 1 - t))) for t in range(T)] for m in range(M + 1)])
Wz, Wx, We, Wg = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
bk, be = rng.normal(size=d), rng.normal(size=d)
a_k = 1.0 + 0.05 * np.arange(LEVELS + 1)
b_k = 0.2 + 0.1 * np.arange(LEVELS + 1)
s_k = 0.05 * np.arange(LEVELS + 1)
x_init = rng.normal(size=(T, d))          # pure noise
w_noise = rng.normal(size=(M, T, d))
z_start = np.zeros(d)


def df_cell(z_left, x_own, k_now, k_next, w):
    z = np.tanh(Wz @ z_left + Wx @ x_own + bk * k_now)
    eps = Wg @ z + We @ x_own + be * k_next
    return z, a_k[k_next] * (x_own - b_k[k_next] * eps) + s_k[k_next] * w


def df_sampler():
    x = x_init.copy()
    for m in range(M - 1, -1, -1):
        z = z_start
        for t in range(T):
            z, x[t] = df_cell(z, x[t], K_sched[m + 1, t], K_sched[m, t], w_noise[m, t])
    return x


direct = df_sampler()
# cell state u = (z, x); pass k of the template is row m = M - 1 - k.
tmpl = run_grid(
    T, M,
    F=lambda i, same, prev, c: df_cell(same, prev, *c),
    read_same=lambda i, cur: cur[i - 1][0] if i > 0 else z_start,     # z of the previous token, this row
    read_prev=lambda i, prev: prev[i][1],                            # the cell's own x, previous row
    cond=lambda i, k: (K_sched[M - k, i], K_sched[M - 1 - k, i], w_noise[M - 1 - k, i]),
    S=[[1] * T] * M, u0=[(None, x_init[i]) for i in range(T)])
report("A1 template == diffusion forcing sampling, Algorithm 2 (token x pass)",
       np.abs(np.array([u[1] for u in tmpl[-1]]) - direct).max())

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
#     the cell below only, previous-pass read = own value only, the level given
#     as an input) and the looped network is the sampler of A1 with layers in
#     the place of tokens.
# ---------------------------------------------------------------------------
def loop_net_reduced():
    own = x_init.copy()                    # one stored value per layer, kept between loops
    for r in range(M):
        below = z_start
        for l in range(T):
            below, own[l] = df_cell(below, own[l], K_sched[M - r, l], K_sched[M - 1 - r, l], w_noise[M - 1 - r, l])
    return own


report("A3 reduced looped network == diffusion forcing sampler", np.abs(loop_net_reduced() - df_sampler()).max())

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

# ---------------------------------------------------------------------------
# E1. Coarse to fine, the Inkling 2 Mini wiring (its reference implementation
#     in transformers, modular_inkling2_loop.py).  4 blocks of NLB layers, a
#     layer = two sublayers.  Every sublayer reads a softmax mix over the
#     RMS-normalised sources [embedding, finished blocks, its block's running
#     sum].  Blocks 2-3 run 3 times; a later pass restarts from the state after
#     block 1, and each of its sublayers also reads a message, a softmax mix
#     over the previous pass's blocks 2 and 3, merged by a low-rank keep/gain
#     gate.  Block 4 follows the last pass; an output mix reads everything.
#
#     The same network as the grid recurrence with one cell per sublayer:
#       u[i, k]          the sublayer's output
#       A_same[j -> i]   alpha_b / RMS(v_b), b = the block of j: one weight
#                        shared by every sublayer of a source block
#       A_prev[j -> i]   beta_b / RMS(v_b of the previous pass), likewise
#       F_i              the gate, then the sublayer
#       S                block 1 in pass 1 only, block 4 in the last pass only
# ---------------------------------------------------------------------------
NBK, NLB, PASSES, RANK = 4, 2, 3, 3
SPB = 2 * NLB                                   # sublayers per block
NS = NBK * SPB
CORE = (1, 2)
Ws = rng.normal(size=(NS, d, d)) / np.sqrt(d)
q_mix = rng.normal(size=(NS, d)) * 2
u_msg = rng.normal(size=(NS, d)) * 2
P_gate = rng.normal(size=(NS, RANK, 2 * d)) / np.sqrt(2 * d)
F_gate = rng.normal(size=(NS, d, RANK))
G_gate = rng.normal(size=(NS, d, RANK))
q_out = rng.normal(size=d) * 2


def rms(x):
    return np.sqrt((x * x).mean(-1) + 1e-12)


def mix_weights(q, srcs):
    return softmax(rms_norm(np.stack(srcs)) @ q / np.sqrt(d))


def mix(q, srcs):
    return mix_weights(q, srcs) @ rms_norm(np.stack(srcs))        # keys and values both normalised


def gate(s, x, m):
    z = P_gate[s] @ np.concatenate([rms_norm(x), rms_norm(m)])
    return (1 - np.tanh(F_gate[s] @ z)) * x + np.tanh(G_gate[s] @ z) * m


def sublayer(s, x):
    return np.tanh(Ws[s] @ rms_norm(x))


def run_block(b, finished, prev_core):
    partial = None
    for s in range(b * SPB, (b + 1) * SPB):
        x = mix(q_mix[s], finished if partial is None else finished + [partial])
        if prev_core is not None:
            x = gate(s, x, mix(u_msg[s], prev_core))
        o = sublayer(s, x)
        partial = o if partial is None else partial + o
    return partial


def fine_network():
    b0 = run_block(0, [emb], None)
    prev = None
    for _ in range(PASSES):
        b1 = run_block(1, [emb, b0], prev)                       # restart: sources are emb and block 1 again
        b2 = run_block(2, [emb, b0, b1], prev)
        prev = [b1, b2]
    b3 = run_block(3, [emb, b0, b1, b2], None)
    return np.stack([b0, b1, b2, b3]), mix(q_out, [emb, b0, b1, b2, b3])


def cells_of(b):
    return range(b * SPB, (b + 1) * SPB)


def weighted_sum_of_cells(values, groups, weights):
    """sum_j A[j] values[j] with A[j] = weight of j's group / RMS of the group's sum."""
    total = np.zeros(d)
    for group, w in zip(groups, weights):
        v = sum(values[j] for j in group)
        for j in group:
            total = total + (w / rms(v)) * values[j]
    return total


def read_same_cells(i, cur):
    b = i // SPB
    groups = [list(cells_of(g)) for g in range(b)] + ([list(range(b * SPB, i))] if i % SPB else [])
    sums = [emb] + [sum(cur[j] for j in g) for g in groups]
    alpha = mix_weights(q_mix[i], sums)
    return alpha[0] * rms_norm(emb) + weighted_sum_of_cells(cur, groups, alpha[1:])


def read_prev_cells(i, prev):
    if i // SPB not in CORE or prev[CORE[0] * SPB] is None:
        return None
    groups = [list(cells_of(g)) for g in CORE]
    beta = mix_weights(u_msg[i], [sum(prev[j] for j in g) for g in groups])
    return weighted_sum_of_cells(prev, groups, beta)


S_fine = [[1 if (b in CORE or (b == 0 and k == 0) or (b == NBK - 1 and k == PASSES - 1)) else 0
           for b in range(NBK) for _ in range(SPB)] for k in range(PASSES)]
hist = run_grid(
    NS, PASSES,
    F=lambda i, same, prev, c: sublayer(i, same if prev is None else gate(i, same, prev)),
    read_same=read_same_cells, read_prev=read_prev_cells, cond=lambda i, k: None,
    S=S_fine, u0=[None] * NS)
cells = hist[-1]
blocks_tmpl = np.stack([sum(cells[j] for j in cells_of(b)) for b in range(NBK)])
blocks_fine, out_fine = fine_network()
out_tmpl = mix(q_out, [emb, *blocks_tmpl])
report("E1 sublayer-level looped block AttnRes == grid recurrence, block sums",
       np.abs(blocks_tmpl - blocks_fine).max())
report("E1 the same, output mix", np.abs(out_tmpl - out_fine).max())

# ---------------------------------------------------------------------------
# E2. The LoopRes read (loopres/kernels.py, static_read): one softmax over the
#     prelude output, the previous step's blocks, this step's finished blocks
#     and the running sum; a source's logit is key . query + value_query .
#     rms(source), with a key per (step, source) and queries per step.  LoRM's
#     AttnRes with feedback also puts both passes' sources in one softmax.
#     It is the grid recurrence when a read returns its unnormalised sum and
#     its mass, F divides, and the reads take the step index: the two reads
#     share a normaliser and the condition enters the read itself.
# ---------------------------------------------------------------------------
LL, RR = 4, 3
prelude_out = rng.normal(size=d)
Wj = rng.normal(size=(LL, d, d)) / np.sqrt(d)
key_prelude = rng.normal(size=d)
key_src = rng.normal(size=(RR, LL, d))            # one static key per (step, source layer)
query_k = rng.normal(size=(RR, LL, d)) * 0.5      # per step
query_v = rng.normal(size=(RR, LL, d)) * 0.5


def joint_read(step, l, srcs_keys):
    logits = np.array([key @ query_k[step, l] + query_v[step, l] @ rms_norm(v) for v, key in srcs_keys])
    return softmax(logits) @ np.stack([v for v, _ in srcs_keys])            # raw values are mixed


def loopres_layers(step, cur):
    return [(v, key_src[step, l]) for l, v in enumerate(cur)]


def loopres_direct():
    prev, out = [], []
    for step in range(RR):
        cur = []
        for l in range(LL):
            srcs = [(prelude_out, key_prelude)] + prev + loopres_layers(step, cur)
            cur.append(np.tanh(Wj[l] @ rms_norm(joint_read(step, l, srcs))))
        prev = loopres_layers(step, cur)
        out.append(list(cur))
    return np.array(out)


def mass_and_sum(step, l, srcs_keys):
    if not srcs_keys:
        return np.zeros(d), 0.0
    logits = np.array([key @ query_k[step, l] + query_v[step, l] @ rms_norm(v) for v, key in srcs_keys])
    e = np.exp(logits)
    return e @ np.stack([v for v, _ in srcs_keys]), e.sum()


def f_joint(i, same, prev, step):
    (s1, m1), (s2, m2) = same, prev
    return np.tanh(Wj[i] @ rms_norm((s1 + s2) / (m1 + m2)))


tmpl = run_grid(
    LL, RR, F=f_joint,
    read_same=lambda i, cur, step: mass_and_sum(step, i, [(prelude_out, key_prelude)] + loopres_layers(step, cur)),
    read_prev=lambda i, prev, step: mass_and_sum(step, i, [] if prev[0] is None else loopres_layers(step - 1, prev)),
    cond=lambda i, k: k, S=[[1] * LL] * RR, u0=[None] * LL, cond_in_reads=True)
report("E2 one softmax over both passes' sources == grid recurrence with a shared normaliser",
       np.abs(np.array(tmpl) - loopres_direct()).max())

if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("ALL CHECKS PASS")
