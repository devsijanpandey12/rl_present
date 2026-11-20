"""
Next-Gen Midnight Asphalt x Autonomous Telemetry Presentation Builder
Generates:
  1. Exact 500-state Taxi-v4 Value Iteration policy & V*(s) JSON (for live web simulator)
  2. Bespoke dark-mode high-DPI figures (matching exact notebook results, seed=42)
  3. Glowing dark-mode equations, PPO clipping diagram, SAC vs eps-greedy chart, backdrops, QR
  4. Animated 15-slide Midnight-Telemetry .pptx with native OpenXML transitions (Google Slides ready!)

Team:
  - Sijan Pandey   (Roll No. 11)
  - Ashim Shrestha (Roll No. 22)
  - Nishan         (Roll No. TBD)
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import qrcode
from PIL import Image, ImageDraw

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls

# ---------------------------------------------------------------------------
# Directories & Palette ("Midnight Asphalt x Autonomous Telemetry HUD")
# ---------------------------------------------------------------------------
ROOT = r"c:\Users\sijan\Desktop\rl_mini_project"
ORIG_FIG_DIR = os.path.join(ROOT, "figures")
OUT_DIR = os.path.join(ROOT, "presentation2")
ASSET_DIR = os.path.join(OUT_DIR, "assets")
os.makedirs(ASSET_DIR, exist_ok=True)
OUT_PPTX = os.path.join(OUT_DIR, "RL_Taxi_v4_NextGen_GoogleSlides.pptx")

# Dark Next-Gen Colors (PPTX RGBColor)
BG_DARK    = RGBColor(0x07, 0x0B, 0x14)  # Obsidian asphalt
CARD_DARK  = RGBColor(0x11, 0x18, 0x27)  # Frosted dark panel
CARD_ELEV  = RGBColor(0x16, 0x20, 0x33)  # Elevated slate surface
BORDER_DIM = RGBColor(0x26, 0x34, 0x4E)  # Luminous hairline border
TXT_MAIN   = RGBColor(0xF8, 0xFA, 0xFC)  # Crisp off-white
TXT_SUB    = RGBColor(0xCB, 0xD5, 0xE1)  # Cool silver
TXT_MUTED  = RGBColor(0x88, 0x99, 0xB4)  # Muted telemetry label

GOLD       = RGBColor(0xFA, 0xCC, 0x15)  # Taxi Gold / SARSA
GOLD_DARK  = RGBColor(0x2D, 0x24, 0x0D)
CYAN       = RGBColor(0x38, 0xBD, 0xF8)  # Electric Cyan / DQN
CYAN_DARK  = RGBColor(0x0C, 0x24, 0x38)
VIOLET     = RGBColor(0xA7, 0x8B, 0xFA)  # Hyper Violet / PPO
VIOLET_DK  = RGBColor(0x1E, 0x16, 0x3A)
EMERALD    = RGBColor(0x34, 0xD3, 0x99)  # Neon Emerald / SAC
EMERALD_DK = RGBColor(0x09, 0x2A, 0x22)
ROSE       = RGBColor(0xFB, 0x71, 0x85)  # Alert Rose / Penalty
ROSE_DK    = RGBColor(0x31, 0x11, 0x1B)

FONT_HEAD = "Segoe UI"
FONT_BODY = "Segoe UI"
FONT_MONO = "Consolas"

# Copy animated GIF into presentation2/assets so presentation2 is self-contained
shutil.copy2(
    os.path.join(ORIG_FIG_DIR, "fig5_taxi_animation_comparison.gif"),
    os.path.join(ASSET_DIR, "fig5_taxi_animation_comparison.gif"),
)

# ---------------------------------------------------------------------------
# 1. Initialize Taxi-v4, Compute Exact Policy/V*(s) JSON & Run Exact Benchmarks
# ---------------------------------------------------------------------------
print("[1/5] Running exact Taxi-v4 solver & notebook benchmarks (seed=42)...")
env = gym.make("Taxi-v4")
n_states = env.observation_space.n
n_actions = env.action_space.n
P_table = env.unwrapped.P


def decode_taxi_state(s):
    d = s % 4; s //= 4
    p = s % 5; s //= 5
    c = s % 5; r = s // 5
    return r, c, p, d


def encode_taxi_state(r, c, p, d):
    return ((r * 5 + c) * 5 + p) * 4 + d


def rolling_mean(arr, window=50):
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().values


# Exact Value Iteration (gamma = 0.95) for live interactive web simulator
V_exact = np.zeros(n_states, dtype=np.float64)
Q_exact = np.zeros((n_states, n_actions), dtype=np.float64)
gamma_vi = 0.95
for _ in range(300):
    V_new = np.zeros_like(V_exact)
    for s in range(n_states):
        for a in range(n_actions):
            _, ns, r, term = P_table[s][a][0]
            Q_exact[s, a] = r + (0.0 if term else gamma_vi * V_exact[ns])
        V_new[s] = np.max(Q_exact[s])
    if np.max(np.abs(V_new - V_exact)) < 1e-8:
        V_exact = V_new
        break
    V_exact = V_new

pi_exact = [int(np.argmax(Q_exact[s])) for s in range(n_states)]
v_exact_list = [round(float(v), 2) for v in V_exact]

with open(os.path.join(ASSET_DIR, "taxi_v4_exact.json"), "w", encoding="utf-8") as f:
    json.dump({"pi": pi_exact, "v": v_exact_list}, f, separators=(",", ":"))


# Exact Notebook Algorithms (Matching README numbers: 596, 808, 944, 1023)
def train_sarsa(episodes=2500, alpha=0.25, gamma=0.95, eps_start=1.0, eps_decay=0.996, eps_min=0.01, seed=42):
    rng = np.random.RandomState(seed)
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    eps = eps_start
    rews, stps, ills = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

    def select_a(s, e):
        if rng.rand() < e:
            return rng.randint(n_actions)
        q_row = Q[s]
        best = np.flatnonzero(np.isclose(q_row, np.max(q_row)))
        return int(rng.choice(best))

    for ep in range(episodes):
        s = rng.randint(n_states)
        _, _, p_c, d_c = decode_taxi_state(s)
        while p_c == 4 or p_c == d_c:
            s = rng.randint(n_states)
            _, _, p_c, d_c = decode_taxi_state(s)
        a = select_a(s, eps)
        er, es, ei = 0.0, 0, 0
        for _ in range(200):
            _, sn, r, term = P_table[s][a][0]
            an = select_a(sn, eps)
            Q[s, a] += alpha * (r + (0.0 if term else gamma * Q[sn, an]) - Q[s, a])
            er += r; es += 1; ei += (1 if r == -10 else 0)
            s, a = sn, an
            if term:
                break
        rews[ep], stps[ep], ills[ep] = er, es, ei
        eps = max(eps_min, eps * eps_decay)
    return Q, {"rewards": rews, "steps": stps, "illegal": ills}


def train_dqn(episodes=2500, lr=0.30, gamma=0.95, eps_start=1.0, eps_decay=0.9955, eps_min=0.01,
              buffer_capacity=12000, batch_size=32, target_sync_steps=150, seed=42):
    rng = np.random.RandomState(seed)
    W_online = np.zeros((n_states, n_actions), dtype=np.float64)
    W_target = W_online.copy()
    buf_s = np.zeros(buffer_capacity, dtype=np.int32)
    buf_a = np.zeros(buffer_capacity, dtype=np.int32)
    buf_r = np.zeros(buffer_capacity, dtype=np.float64)
    buf_s2 = np.zeros(buffer_capacity, dtype=np.int32)
    buf_d = np.zeros(buffer_capacity, dtype=np.float64)
    buf_ptr, buf_size, total_steps = 0, 0, 0
    epsilon = eps_start
    rewards_hist, steps_hist, illegal_hist = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

    for ep in range(episodes):
        s = rng.randint(n_states)
        _, _, p_c, d_c = decode_taxi_state(s)
        while p_c == 4 or p_c == d_c:
            s = rng.randint(n_states)
            _, _, p_c, d_c = decode_taxi_state(s)
        ep_reward, ep_steps, ep_illegal = 0.0, 0, 0
        for _ in range(200):
            if rng.rand() < epsilon:
                a = rng.randint(n_actions)
            else:
                q_vals = W_online[s]
                mx = np.max(q_vals)
                a = int(rng.choice(np.flatnonzero(np.isclose(q_vals, mx))))
            _, s_next, reward, terminated = P_table[s][a][0]
            buf_s[buf_ptr] = s; buf_a[buf_ptr] = a; buf_r[buf_ptr] = reward
            buf_s2[buf_ptr] = s_next; buf_d[buf_ptr] = 1.0 if terminated else 0.0
            buf_ptr = (buf_ptr + 1) % buffer_capacity
            buf_size = min(buf_size + 1, buffer_capacity)
            total_steps += 1
            best_next_t = 0.0 if terminated else np.max(W_target[s_next])
            W_online[s, a] += lr * (reward + gamma * best_next_t - W_online[s, a])
            if buf_size >= batch_size and total_steps % 4 == 0:
                idx = rng.randint(0, buf_size, size=batch_size)
                b_s, b_a, b_r, b_s2, b_d = buf_s[idx], buf_a[idx], buf_r[idx], buf_s2[idx], buf_d[idx]
                targets = b_r + gamma * (1.0 - b_d) * np.max(W_target[b_s2], axis=1)
                td_errors = targets - W_online[b_s, b_a]
                np.add.at(W_online, (b_s, b_a), 0.15 * lr * td_errors)
            if total_steps % target_sync_steps == 0:
                W_target[:] = W_online
            ep_reward += reward; ep_steps += 1; ep_illegal += (1 if reward == -10 else 0)
            s = s_next
            if terminated:
                break
        rewards_hist[ep], steps_hist[ep], illegal_hist[ep] = ep_reward, ep_steps, ep_illegal
        epsilon = max(eps_min, epsilon * eps_decay)
    return W_online, {"rewards": rewards_hist, "steps": steps_hist, "illegal": illegal_hist}


def train_ppo(episodes=2500, actor_lr=0.35, critic_lr=0.30, gamma=0.95, clip_eps=0.20,
              ppo_epochs=3, entropy_coef=0.015, seed=42):
    rng = np.random.RandomState(seed)
    logits = np.zeros((n_states, n_actions), dtype=np.float64)
    V_critic = np.zeros(n_states, dtype=np.float64)
    Q_critic = np.zeros((n_states, n_actions), dtype=np.float64)
    rewards_hist, steps_hist, illegal_hist = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

    def get_probs(s):
        z = logits[s] - np.max(logits[s])
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    for ep in range(episodes):
        s = rng.randint(n_states)
        _, _, p_c, d_c = decode_taxi_state(s)
        while p_c == 4 or p_c == d_c:
            s = rng.randint(n_states)
            _, _, p_c, d_c = decode_taxi_state(s)
        traj_s, traj_a, traj_old_p, traj_adv = [], [], [], []
        ep_reward, ep_steps, ep_illegal = 0.0, 0, 0
        for _ in range(200):
            probs = get_probs(s)
            a = int(rng.choice(n_actions, p=probs))
            _, s_next, reward, terminated = P_table[s][a][0]
            next_v = 0.0 if terminated else np.max(Q_critic[s_next])
            Q_critic[s, a] += critic_lr * (reward + gamma * next_v - Q_critic[s, a])
            V_critic[s] = np.sum(probs * Q_critic[s])
            advantage = Q_critic[s, a] - V_critic[s]
            traj_s.append(s); traj_a.append(a); traj_old_p.append(probs[a]); traj_adv.append(advantage)
            ep_reward += reward; ep_steps += 1; ep_illegal += (1 if reward == -10 else 0)
            s = s_next
            if terminated:
                break
        for _ in range(ppo_epochs):
            for s_t, a_t, old_p, adv_t in zip(traj_s, traj_a, traj_old_p, traj_adv):
                curr_probs = get_probs(s_t)
                ratio = curr_probs[a_t] / (old_p + 1e-8)
                clipped_Flag = (adv_t > 0 and ratio > 1.0 + clip_eps) or (adv_t < 0 and ratio < 1.0 - clip_eps)
                if not clipped_Flag:
                    grad = -curr_probs; grad[a_t] += 1.0
                    ent_grad = -curr_probs * (np.log(curr_probs + 1e-8) + 1.0)
                    logits[s_t] += actor_lr * (adv_t * grad + entropy_coef * ent_grad)
        rewards_hist[ep], steps_hist[ep], illegal_hist[ep] = ep_reward, ep_steps, ep_illegal
    return logits, {"rewards": rewards_hist, "steps": steps_hist, "illegal": illegal_hist}


def train_sac(episodes=2500, lr=0.30, gamma=0.95, tau_start=1.2, tau_decay=0.996, tau_min=0.08, seed=42):
    rng = np.random.RandomState(seed)
    Q1 = np.zeros((n_states, n_actions), dtype=np.float64)
    Q2 = np.zeros((n_states, n_actions), dtype=np.float64)
    tau = tau_start
    rews, stps, ills = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

    def soft_pi(s, t):
        qm = np.minimum(Q1[s], Q2[s])
        sc = (qm - np.max(qm)) / t
        eq = np.exp(sc)
        return eq / np.sum(eq)

    for ep in range(episodes):
        s = rng.randint(n_states)
        _, _, p_c, d_c = decode_taxi_state(s)
        while p_c == 4 or p_c == d_c:
            s = rng.randint(n_states)
            _, _, p_c, d_c = decode_taxi_state(s)
        er, es, ei = 0.0, 0, 0
        for _ in range(200):
            probs = soft_pi(s, tau)
            a = int(rng.choice(n_actions, p=probs))
            _, sn, r, term = P_table[s][a][0]
            if term:
                vsn = 0.0
            else:
                np_ = soft_pi(sn, tau)
                nqm = np.minimum(Q1[sn], Q2[sn])
                vsn = float(np.sum(np_ * (nqm - tau * np.log(np_ + 1e-10))))
            tgt = r + gamma * vsn
            Q1[s, a] += lr * (tgt - Q1[s, a])
            Q2[s, a] += lr * (tgt - Q2[s, a])
            er += r; es += 1; ei += (1 if r == -10 else 0)
            s = sn
            if term:
                break
        rews[ep], stps[ep], ills[ep] = er, es, ei
        tau = max(tau_min, tau * tau_decay)
    return np.minimum(Q1, Q2), {"rewards": rews, "steps": stps, "illegal": ills}


Q_sarsa, sarsa_hist = train_sarsa()
Q_dqn, dqn_hist = train_dqn()
ppo_logits, ppo_hist = train_ppo()
Q_sac, sac_hist = train_sac()

# Export real 60-episode rolling curves (every 10th episode) for the animated web "Learning Race"
_curves = {}
for _nm, _h in [("sarsa", sarsa_hist), ("dqn", dqn_hist), ("ppo", ppo_hist), ("sac", sac_hist)]:
    _curves[_nm] = {k: [round(float(x), 2) for x in rolling_mean(_h[k], 60)[::10]] for k in ("rewards", "steps", "illegal")}
    _curves[_nm]["total_illegal"] = int(np.sum(_h["illegal"]))
with open(os.path.join(ASSET_DIR, "curves.json"), "w", encoding="utf-8") as f:
    json.dump(_curves, f, separators=(",", ":"))
print("   exported curves.json ->", {k: v["total_illegal"] for k, v in _curves.items()})

# ---------------------------------------------------------------------------
# 2. Generate Bespoke Dark-Mode ("Midnight Asphalt") Figures & Visual Assets
# ---------------------------------------------------------------------------
print("[2/5] Rendering bespoke dark-mode figures, equations, backdrops & QR...")

DARK_BG = "#0B1120"
DARK_PANEL = "#111827"
DARK_GRID = "#1E293B"
DARK_EDGE = "#334155"
DARK_TXT = "#F1F5F9"
DARK_MUTED = "#94A3B8"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_PANEL,
    "axes.edgecolor": DARK_EDGE,
    "axes.labelcolor": DARK_TXT,
    "axes.titlecolor": DARK_TXT,
    "xtick.color": DARK_MUTED,
    "ytick.color": DARK_MUTED,
    "text.color": DARK_TXT,
    "axes.grid": True,
    "grid.color": DARK_GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.9,
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def make_midnight_backdrop(filename, hero=False):
    """Architectural subtle coordinate grid + ambient radial glow background."""
    w, h = 1920, 1080
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :] = [7, 11, 20]  # #070B14

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Ambient cyan glow top-right & amber glow bottom-left
    g1 = np.exp(-((xx - w * 0.85) ** 2 / (2 * 480 ** 2) + (yy - h * 0.12) ** 2 / (2 * 380 ** 2)))
    g2 = np.exp(-((xx - w * 0.12) ** 2 / (2 * 520 ** 2) + (yy - h * 0.88) ** 2 / (2 * 400 ** 2)))
    if hero:
        g3 = np.exp(-((xx - w * 0.50) ** 2 / (2 * 600 ** 2) + (yy - h * 0.45) ** 2 / (2 * 450 ** 2)))
        base[:, :, 0] += g1 * 18 + g2 * 34 + g3 * 12
        base[:, :, 1] += g1 * 48 + g2 * 24 + g3 * 24
        base[:, :, 2] += g1 * 76 + g2 * 10 + g3 * 48
    else:
        base[:, :, 0] += g1 * 10 + g2 * 16
        base[:, :, 1] += g1 * 26 + g2 * 14
        base[:, :, 2] += g1 * 42 + g2 * 8

    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")
    # Subtle coordinate grid
    step = 96
    alpha = 20 if hero else 11
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=(148, 163, 184, alpha), width=1)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=(148, 163, 184, alpha), width=1)

    if hero:
        # Glowing route lines hinting at autonomous taxi pathing
        pts = [(1180, 180), (1180, 520), (1460, 520), (1460, 820), (1760, 820)]
        draw.line(pts, fill=(56, 189, 248, 55), width=6)
        draw.line(pts, fill=(56, 189, 248, 140), width=2)
        for px, py in pts:
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(250, 204, 21, 180))

    path = os.path.join(ASSET_DIR, filename)
    img.save(path, "PNG")
    return path


bg_hero_path = make_midnight_backdrop("dark_bg_hero.png", hero=True)
bg_slide_path = make_midnight_backdrop("dark_bg_slide.png", hero=False)


def render_dark_equation(latex_str, filename, fontsize=26, color="#F8FAFC"):
    path = os.path.join(ASSET_DIR, filename)
    fig = plt.figure(figsize=(8.6, 0.92), dpi=300)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize, color=color, ha="center", va="center")
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


eq_state  = render_dark_equation(r"$s_{\mathrm{id}} = ((5r + c) \times 5 + p) \times 4 + d \;\in\; \{0, \dots, 499\}$", "dark_eq_state.png", 25, "#38BDF8")
eq_sarsa  = render_dark_equation(r"$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[\, R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \,\right]$", "dark_eq_sarsa.png", 25, "#FACC15")
eq_dqn    = render_dark_equation(r"$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d)\sim\mathcal{D}}\!\left[\,\left(r + \gamma(1-d)\max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\,\right]$", "dark_eq_dqn.png", 23, "#38BDF8")
eq_ppo    = render_dark_equation(r"$L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\,\min\!\left(r_t(\theta)\hat{A}_t,\; \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat{A}_t\right) + c_{\mathrm{ent}}\mathcal{H}(\pi_\theta)\,\right]$", "dark_eq_ppo.png", 21, "#C4B5FD")
eq_sac_v  = render_dark_equation(r"$V_{\mathrm{soft}}(s') = \sum_{a'} \pi(a'|s')\left(\min_{i\in\{1,2\}} Q_i(s',a') - \tau \log \pi(a'|s')\right)$", "dark_eq_sac_v.png", 21, "#34D399")
eq_sac_pi = render_dark_equation(r"$\pi(a \mid s) = \frac{\exp\!\left(\min_{i\in\{1,2\}} Q_i(s,a)\, /\, \tau\right)}{\sum_{a'} \exp\!\left(\min_{i\in\{1,2\}} Q_i(s,a')\, /\, \tau\right)}$", "dark_eq_sac_pi.png", 21, "#F8FAFC")


# Dark Figure 1: Taxi-v4 Telemetry Map
def render_dark_fig1():
    stations = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}
    s_cols = {0: "#FB7185", 1: "#34D399", 2: "#FACC15", 3: "#38BDF8"}
    s_lbls = {0: "R", 1: "G", 2: "Y", 3: "B"}
    walls = {(0, 1), (1, 1), (3, 0), (4, 0), (3, 2), (4, 2)}

    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(13.6, 5.1), gridspec_kw={"width_ratios": [1.05, 1.08]})
    fig.patch.set_facecolor(DARK_BG)
    ax_map.set_facecolor("#0F172A")
    ax_map.set_xlim(0, 5); ax_map.set_ylim(0, 5); ax_map.invert_yaxis()
    ax_map.set_xticks(np.arange(0.5, 5.5, 1.0)); ax_map.set_xticklabels([f"Col {c}" for c in range(5)], fontweight="bold", color="#CBD5E1")
    ax_map.set_yticks(np.arange(0.5, 5.5, 1.0)); ax_map.set_yticklabels([f"Row {r}" for r in range(5)], fontweight="bold", color="#CBD5E1")
    ax_map.set_title("Gymnasium Taxi-v4 Telemetry Map (5×5)", pad=12, color="#F8FAFC")
    ax_map.grid(False)

    for r in range(5):
        for c in range(5):
            ax_map.add_patch(mpatches.Rectangle((c, r), 1, 1, facecolor="#0F172A", edgecolor="#1E293B", lw=1.3))
            ax_map.text(c + 0.12, r + 0.22, f"({r},{c})", fontsize=8, color="#475569", family="monospace")

    for idx, (sr, sc) in stations.items():
        col = s_cols[idx]
        ax_map.add_patch(mpatches.Rectangle((sc + 0.06, sr + 0.06), 0.88, 0.88, facecolor=col, alpha=0.18, edgecolor=col, lw=2.4))
        ax_map.text(sc + 0.5, sr + 0.52, s_lbls[idx], fontsize=20, fontweight="bold", color=col, ha="center", va="center")

    for (wr, wc) in walls:
        ax_map.plot([wc + 1, wc + 1], [wr + 0.05, wr + 0.95], color="#F97316", lw=7.0, solid_capstyle="round", alpha=0.35)
        ax_map.plot([wc + 1, wc + 1], [wr + 0.05, wr + 0.95], color="#FB923C", lw=3.0, linestyle="--")

    taxi = mpatches.FancyBboxPatch((2.16, 2.22), 0.68, 0.56, boxstyle="round,pad=0.05", facecolor="#FACC15", edgecolor="#FEF08A", lw=2)
    ax_map.add_patch(taxi)
    ax_map.text(2.5, 2.5, "TAXI", fontsize=9.5, fontweight="bold", color="#070B14", ha="center", va="center")

    for sp in ax_map.spines.values():
        sp.set_edgecolor("#38BDF8"); sp.set_linewidth(1.8)

    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1); ax_info.axis("off")
    box = mpatches.FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.02", facecolor="#111827", edgecolor="#334155", lw=1.5)
    ax_info.add_patch(box)
    txt = (
        "GYMNASIUM TAXI-V4 SPECIFICATION\n"
        "───────────────────────────────────────────────────\n\n"
        "• States  |S| = 500  (25 cells × 5 pass × 4 dest)\n"
        "• Actions |A| = 6    (S, N, E, W, Pickup, Dropoff)\n"
        "• Step Reward        : -1 per time step\n"
        "• Successful Dropoff : +20 (Terminal Goal)\n"
        "• Illegal P/D Action : -10 Penalty\n\n"
        "ALGORITHMS COMPARED (CONTROLLED ABLATION)\n"
        "───────────────────────────────────────────────────\n"
        "1. SARSA : On-Policy Tabular Temporal-Difference\n"
        "2. DQN   : Off-Policy Q-Learning + Replay & Target\n"
        "3. PPO   : On-Policy Clipped Actor-Critic\n"
        "4. SAC   : Off-Policy Max-Entropy Twin Critics"
    )
    ax_info.text(0.05, 0.50, txt, fontsize=9.5, family="monospace", color="#E2E8F0", va="center", linespacing=1.48)
    plt.tight_layout()
    p = os.path.join(ASSET_DIR, "dark_fig1_layout.png")
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return p


# Dark Figure 2: 4-Panel Benchmark
def render_dark_fig2():
    algos = {
        "SARSA (On-Policy TD)":       (sarsa_hist, "#FACC15"),
        "DQN (Off-Policy Replay)":    (dqn_hist,   "#38BDF8"),
        "PPO (Clipped Actor-Critic)": (ppo_hist,   "#A78BFA"),
        "SAC (Max-Entropy Soft AC)":  (sac_hist,   "#34D399"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.6))
    fig.patch.set_facecolor(DARK_BG)
    ep_x = np.arange(1, 2501)

    ax = axes[0, 0]
    for name, (h, col) in algos.items():
        ax.plot(ep_x, rolling_mean(h["rewards"], 60), label=name, color=col, lw=2.4)
    ax.axhline(7.9, color="#F8FAFC", linestyle="--", lw=1.3, alpha=0.75, label="Optimal (+7.9)")
    ax.set_title("A. Cumulative Episode Return (60-Ep Moving Avg)")
    ax.set_xlabel("Training Episode"); ax.set_ylabel("Episode Return"); ax.set_ylim(-220, 18)
    ax.legend(loc="lower right", fontsize=9, facecolor="#0F172A", edgecolor="#334155")

    ax = axes[0, 1]
    for name, (h, col) in algos.items():
        ax.plot(ep_x, rolling_mean(h["steps"], 60), label=name, color=col, lw=2.4)
    ax.axhline(13.1, color="#F8FAFC", linestyle="--", lw=1.3, alpha=0.75, label="Optimal Steps (13.1)")
    ax.set_title("B. Steps per Episode to Complete Delivery")
    ax.set_xlabel("Training Episode"); ax.set_ylabel("Steps per Episode")
    ax.legend(loc="upper right", fontsize=9, facecolor="#0F172A", edgecolor="#334155")

    ax = axes[1, 0]
    for name, (h, col) in algos.items():
        ax.plot(ep_x, rolling_mean(h["illegal"], 60), label=name, color=col, lw=2.4)
    ax.set_title("C. Illegal Pickup / Drop-off Penalties (−10) per Episode")
    ax.set_xlabel("Training Episode"); ax.set_ylabel("Illegal Actions / Ep")
    ax.legend(loc="upper right", fontsize=9, facecolor="#0F172A", edgecolor="#334155")

    ax = axes[1, 1]
    short = ["SARSA", "DQN", "PPO", "SAC"]
    cols = ["#FACC15", "#38BDF8", "#A78BFA", "#34D399"]
    convs = []
    for _, (h, _) in algos.items():
        sm = rolling_mean(h["rewards"], 50)
        idx = np.where(sm >= 5.0)[0]
        convs.append(int(idx[0]) if len(idx) > 0 else 2500)
    bars = ax.bar(short, convs, color=cols, edgecolor="#F8FAFC", lw=1.1, width=0.52)
    for b, val in zip(bars, convs):
        ax.text(b.get_x() + b.get_width() / 2, val + 26, f"{val} eps", ha="center", va="bottom",
                fontweight="bold", fontsize=11, color="#F8FAFC")
    ax.set_title("D. Sample Efficiency: Episodes to Reach +5.0 Return")
    ax.set_ylabel("Episodes Required (Lower is Faster)")
    ax.set_ylim(0, max(convs) * 1.20)

    plt.tight_layout(pad=1.6)
    p = os.path.join(ASSET_DIR, "dark_fig2_comparison.png")
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return p


# Dark Figure 3: Hyperparameter Sensitivity
def render_dark_fig3():
    p = os.path.join(ASSET_DIR, "dark_fig3_sensitivity.png")
    if os.path.exists(p):
        return p
    def run_td(episodes=2000, alpha=0.25, gamma=0.95, eps_start=0.10, eps_decay=1.0, eps_min=0.01, n_seeds=3):
        arr = np.zeros((n_seeds, episodes))
        for si in range(n_seeds):
            _, h = train_sarsa(episodes=episodes, alpha=alpha, gamma=gamma,
                               eps_start=eps_start, eps_decay=eps_decay, eps_min=eps_min, seed=200 + si * 13)
            arr[si] = h["rewards"]
        return np.mean(arr, axis=0)

    a_c = {"α = 0.05 (Slow)": run_td(alpha=0.05), "α = 0.20 (Balanced)": run_td(alpha=0.20),
           "α = 0.50 (Fast)": run_td(alpha=0.50), "α = 0.90 (Overshoot)": run_td(alpha=0.90)}
    g_c = {"γ = 0.50 (Myopic)": run_td(gamma=0.50), "γ = 0.80 (Moderate)": run_td(gamma=0.80),
           "γ = 0.95 (Farsighted)": run_td(gamma=0.95), "γ = 0.99 (Long-Term)": run_td(gamma=0.99)}
    e_c = {"ε = 0.01 (Near-Greedy)": run_td(eps_start=0.01), "ε = 0.10 (Moderate)": run_td(eps_start=0.10),
           "ε = 0.30 (High Explor.)": run_td(eps_start=0.30),
           "Decaying ε (1.0→0.01)": run_td(eps_start=1.0, eps_decay=0.995, eps_min=0.01)}

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))
    fig.patch.set_facecolor(DARK_BG)
    ep_x = np.arange(1, 2001)
    cols = ["#FB7185", "#38BDF8", "#34D399", "#A78BFA"]
    for ax, title, d in [
        (axes[0], "A. Learning Rate Sensitivity (α)", a_c),
        (axes[1], "B. Discount Factor Sensitivity (γ)", g_c),
        (axes[2], "C. Exploration Rate Sensitivity (ε)", e_c),
    ]:
        for (lbl, arr), c in zip(d.items(), cols):
            ax.plot(ep_x, rolling_mean(arr, 60), label=lbl, color=c, lw=2.3)
        ax.set_title(title); ax.set_xlabel("Training Episode"); ax.set_ylabel("Episode Return (60-Ep Avg)")
        ax.set_ylim(-220, 18)
        ax.legend(loc="lower right", fontsize=8.8, facecolor="#0F172A", edgecolor="#334155")

    plt.tight_layout(pad=1.4)
    p = os.path.join(ASSET_DIR, "dark_fig3_sensitivity.png")
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return p


# Dark Figure 4: V*(s) Heatmaps & Trajectory
def render_dark_fig4():
    pass_idx, dest_idx = 2, 1  # Y(4,0) -> G(0,4)
    walls = {(0, 1), (1, 1), (3, 0), (4, 0), (3, 2), (4, 2)}
    stations = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}
    s_cols = {0: "#FB7185", 1: "#34D399", 2: "#FACC15", 3: "#38BDF8"}

    V1, A1 = np.zeros((5, 5)), np.zeros((5, 5), dtype=int)
    V2, A2 = np.zeros((5, 5)), np.zeros((5, 5), dtype=int)
    for r in range(5):
        for c in range(5):
            s1 = encode_taxi_state(r, c, pass_idx, dest_idx)
            V1[r, c], A1[r, c] = np.max(Q_dqn[s1]), np.argmax(Q_dqn[s1])
            s2 = encode_taxi_state(r, c, 4, dest_idx)
            V2[r, c], A2[r, c] = np.max(Q_dqn[s2]), np.argmax(Q_dqn[s2])

    arrows = {0: (0.0, 0.28), 1: (0.0, -0.28), 2: (0.28, 0.0), 3: (-0.28, 0.0)}
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    fig.patch.set_facecolor(DARK_BG)

    def draw_hm(ax, Vm, Am, title, cmap):
        im = ax.imshow(Vm, cmap=cmap, origin="upper", extent=[0, 5, 5, 0], alpha=0.92)
        ax.set_xticks(np.arange(0.5, 5.5, 1.0)); ax.set_xticklabels([f"C{c}" for c in range(5)])
        ax.set_yticks(np.arange(0.5, 5.5, 1.0)); ax.set_yticklabels([f"R{r}" for r in range(5)])
        ax.set_title(title, pad=10); ax.grid(False)
        vmin, vmax = np.min(Vm), np.max(Vm)
        mid = vmin + 0.55 * (vmax - vmin)
        for r in range(5):
            for c in range(5):
                ax.add_patch(mpatches.Rectangle((c, r), 1, 1, fill=False, edgecolor="#070B14", lw=1.3, alpha=0.8))
                ink = "#070B14" if Vm[r, c] > mid else "#F8FAFC"
                ax.text(c + 0.5, r + 0.23, f"{Vm[r, c]:.1f}", ha="center", va="center", fontsize=8.8, fontweight="bold", color=ink)
                act = Am[r, c]
                if act in arrows:
                    dx, dy = arrows[act]
                    ax.annotate("", xy=(c + 0.5 + dx, r + 0.62 + dy), xytext=(c + 0.5 - dx * 0.5, r + 0.62 - dy * 0.5),
                                arrowprops=dict(arrowstyle="->", color=ink, lw=2.3))
                elif act == 4:
                    ax.text(c + 0.5, r + 0.66, "PICKUP", ha="center", va="center", fontsize=8, fontweight="bold", color="#070B14",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FACC15", edgecolor="#FEF08A"))
                elif act == 5:
                    ax.text(c + 0.5, r + 0.66, "DROPOFF", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#070B14",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#34D399", edgecolor="#A7F3D0"))
        for (wr, wc) in walls:
            ax.plot([wc + 1, wc + 1], [wr, wr + 1], color="#FB923C", lw=4.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="#94A3B8")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#CBD5E1")

    draw_hm(axes[0], V1, A1, "Stage 1: Route to Passenger Y(4,0)", "viridis")
    draw_hm(axes[1], V2, A2, "Stage 2: Deliver Passenger to G(0,4)", "inferno")

    # Right panel: Rollout Trajectory
    ax = axes[2]
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.invert_yaxis()
    ax.set_xticks(np.arange(0.5, 5.5, 1.0)); ax.set_xticklabels([f"C{c}" for c in range(5)])
    ax.set_yticks(np.arange(0.5, 5.5, 1.0)); ax.set_yticklabels([f"R{r}" for r in range(5)])
    ax.set_title("Optimal 14-Step Rollout: (0,1) → Y → G", pad=10); ax.grid(False)
    for r in range(5):
        for c in range(5):
            ax.add_patch(mpatches.Rectangle((c, r), 1, 1, facecolor="#0F172A", edgecolor="#1E293B", lw=1.2))
    for idx, (sr, sc) in stations.items():
        col = s_cols[idx]
        lbl = ["R", "G", "Y", "B"][idx]
        ax.add_patch(mpatches.Rectangle((sc + 0.05, sr + 0.05), 0.9, 0.9, facecolor=col, alpha=0.20, edgecolor=col, lw=2.0))
        ax.text(sc + 0.18, sr + 0.24, lbl, fontsize=12, fontweight="bold", color=col)
    for (wr, wc) in walls:
        ax.plot([wc + 1, wc + 1], [wr + 0.04, wr + 0.96], color="#FB923C", lw=5.0)

    # Trace actual optimal path from (0,1), pass=Y(2), dest=G(1)
    s = encode_taxi_state(0, 1, 2, 1)
    s1_pts, s2_pts = [(0, 1)], []
    in_taxi = False
    for _ in range(30):
        a = int(np.argmax(Q_dqn[s]))
        _, sn, _, term = P_table[s][a][0]
        r, c, p, _ = decode_taxi_state(sn)
        if not in_taxi:
            s1_pts.append((r, c))
            if p == 4:
                in_taxi = True
                s2_pts.append((r, c))
        else:
            s2_pts.append((r, c))
        s = sn
        if term:
            break

    for i in range(len(s1_pts) - 1):
        r1, c1 = s1_pts[i]; r2, c2 = s1_pts[i + 1]
        if (r1, c1) != (r2, c2):
            ax.annotate("", xy=(c2 + 0.42, r2 + 0.42), xytext=(c1 + 0.42, r1 + 0.42),
                        arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2.8))
    for i in range(len(s2_pts) - 1):
        r1, c1 = s2_pts[i]; r2, c2 = s2_pts[i + 1]
        if (r1, c1) != (r2, c2):
            ax.annotate("", xy=(c2 + 0.58, r2 + 0.58), xytext=(c1 + 0.58, r1 + 0.58),
                        arrowprops=dict(arrowstyle="->", color="#34D399", lw=2.8))
    ax.plot([], [], color="#38BDF8", lw=3, label="Stage 1: To Pickup (Y)")
    ax.plot([], [], color="#34D399", lw=3, label="Stage 2: To Dropoff (G)")
    ax.legend(loc="lower right", fontsize=8.8, facecolor="#0F172A", edgecolor="#334155")

    plt.tight_layout(pad=1.3)
    p = os.path.join(ASSET_DIR, "dark_fig4_heatmaps.png")
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return p


# Dark SAC vs Epsilon-Greedy Comparison
def render_dark_sac_bars():
    actions = ["South\n(Q=−1)", "North\n(Q=−1)", "East\n(Q=0)", "West\n(Q=−1)", "Illegal P\n(Q=−10)", "Illegal D\n(Q=−10)"]
    qv = np.array([-1.0, -1.0, 0.0, -1.0, -10.0, -10.0])
    p_eg = np.full(6, 0.30 / 6.0); p_eg[2] += 0.70
    z = (qv - np.max(qv)) / 1.0
    p_sac = np.exp(z) / np.sum(np.exp(z))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), dpi=220, sharey=True)
    fig.patch.set_facecolor(DARK_PANEL)
    for ax, probs, col_ok, title in [
        (axes[0], p_eg * 100, "#38BDF8", "ε-Greedy (ε = 0.30)\nStill hits −10 actions 10% of the time"),
        (axes[1], p_sac * 100, "#34D399", "SAC Boltzmann (τ = 1.0)\nCrushes −10 actions to ~0.002%"),
    ]:
        ax.set_facecolor("#0B1120")
        cols = [col_ok] * 4 + ["#FB7185", "#FB7185"]
        bars = ax.bar(range(6), probs, color=cols, edgecolor="#F8FAFC", lw=0.8, width=0.62)
        ax.set_xticks(range(6)); ax.set_xticklabels(actions, fontsize=8.2, fontweight="bold", color="#CBD5E1")
        ax.set_ylim(0, 92); ax.set_title(title, fontsize=10, color="#F8FAFC", pad=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for b, pr in zip(bars, probs):
            lbl = f"{pr:.1f}%" if pr >= 0.1 else "~0%"
            ax.text(b.get_x() + b.get_width() / 2, pr + 2.2, lbl, ha="center", va="bottom",
                    fontsize=8.3, fontweight="bold", color="#F8FAFC")
    axes[0].set_ylabel("Action Prob. (%)", fontsize=9.2, fontweight="bold", color="#CBD5E1")
    plt.tight_layout()
    p = os.path.join(ASSET_DIR, "dark_sac_vs_egreedy.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor=DARK_PANEL)
    plt.close(fig)
    return p


# Dark PPO Clipping Surrogate Visual
def render_dark_ppo_clip():
    r = np.linspace(0.4, 1.6, 300)
    eps = 0.2
    unclipped = r
    clipped = np.minimum(r, np.clip(r, 1 - eps, 1 + eps))

    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=220)
    fig.patch.set_facecolor(DARK_PANEL)
    ax.set_facecolor("#0B1120")
    ax.axvspan(0.8, 1.2, color="#A78BFA", alpha=0.14, label="Safe Trust Region [0.8, 1.2]")
    ax.plot(r, unclipped, color="#64748B", linestyle="--", lw=1.8, label="Unclipped r(θ)·A")
    ax.plot(r, clipped, color="#A78BFA", lw=3.0, label="PPO Clipped Objective (A > 0)")
    ax.axvline(1.2, color="#FB7185", linestyle=":", lw=1.6)
    ax.scatter([1.2], [1.2], color="#FACC15", s=65, zorder=5)
    ax.annotate("Clipped ceiling\nprevents collapse", xy=(1.2, 1.2), xytext=(1.23, 0.85),
                color="#FACC15", fontsize=8.8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#FACC15", lw=1.4))
    ax.set_xlabel("Probability Ratio  r(θ) = π_new / π_old", fontsize=9.2)
    ax.set_ylabel("Surrogate Objective  (A > 0)", fontsize=9.2)
    ax.set_title("PPO Clipped Trust Region (ε = 0.20)", fontsize=10.5, pad=8)
    ax.legend(loc="upper left", fontsize=8.0, facecolor="#111827", edgecolor="#334155")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = os.path.join(ASSET_DIR, "dark_ppo_clip.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor=DARK_PANEL)
    plt.close(fig)
    return p


def render_qr():
    p = os.path.join(ASSET_DIR, "dark_github_qr.png")
    qr = qrcode.QRCode(version=2, box_size=10, border=2)
    qr.add_data("https://github.com/devsijanpandey12/RL_FINAL_MINIPROJECT")
    qr.make(fit=True)
    img = qr.make_image(fill_color="#070B14", back_color="#F8FAFC")
    img.save(p)
    return p


fig1_dark = render_dark_fig1()
fig2_dark = render_dark_fig2()
fig3_dark = render_dark_fig3()
fig4_dark = render_dark_fig4()
sac_bars_dark = render_dark_sac_bars()
ppo_clip_dark = render_dark_ppo_clip()
qr_dark = render_qr()


# ---------------------------------------------------------------------------
# 3. Build 15-Slide Animated Midnight-Telemetry PPTX (Google Slides Optimized!)
# ---------------------------------------------------------------------------
print("[3/5] Building 15-slide Midnight-Telemetry PPTX with native transitions...")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide_transition(slide, kind="fade", spd="med"):
    """Inject native OpenXML <p:transition> (supported by PowerPoint & Google Slides import)."""
    if kind == "push":
        xml = f'<p:transition {nsdecls("p")} spd="{spd}"><p:push dir="l"/></p:transition>'
    elif kind == "wipe":
        xml = f'<p:transition {nsdecls("p")} spd="{spd}"><p:wipe dir="l"/></p:transition>'
    else:
        xml = f'<p:transition {nsdecls("p")} spd="{spd}"><p:fade/></p:transition>'
    slide._element.append(parse_xml(xml))


def add_backdrop(slide, hero=False):
    slide.shapes.add_picture(bg_hero_path if hero else bg_slide_path, Inches(0), Inches(0),
                             width=Inches(13.333), height=Inches(7.5))


def add_card(slide, left, top, w, h, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2, rounded=True):
    st = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(st, Inches(left), Inches(top), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if border:
        shp.line.color.rgb = border; shp.line.width = Pt(border_pt)
    else:
        shp.line.fill.background()
    if rounded:
        try:
            shp.adjustments[0] = 0.06
        except Exception:
            pass
    return shp


def add_glow_bar(slide, left, top, w, h, color):
    return add_card(slide, left, top, w, h, fill=color, border=None, rounded=False)


def add_txt(slide, left, top, w, h, text, size=16, bold=False, color=TXT_MAIN,
            align=PP_ALIGN.LEFT, font=FONT_BODY, valign=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.05); tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.name = font; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return tb


def add_pill(slide, left, top, w, h, text, bg, fg, size=11, border=None):
    shp = add_card(slide, left, top, w, h, fill=bg, border=border or fg, border_pt=1.0, rounded=True)
    try:
        shp.adjustments[0] = 0.45
    except Exception:
        pass
    tf = shp.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.name = FONT_HEAD; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = fg
    return shp


def add_bullets(slide, left, top, w, h, items, size=14.2, accent=CYAN, space_after=9):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
    for i, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after); p.line_spacing = 1.20
        r0 = p.add_run(); r0.text = "▸ "; r0.font.name = FONT_BODY; r0.font.size = Pt(size); r0.font.bold = True; r0.font.color.rgb = accent
        if lead:
            r1 = p.add_run(); r1.text = lead + " "; r1.font.name = FONT_BODY; r1.font.size = Pt(size); r1.font.bold = True; r1.font.color.rgb = TXT_MAIN
        r2 = p.add_run(); r2.text = body; r2.font.name = FONT_BODY; r2.font.size = Pt(size); r2.font.color.rgb = TXT_SUB
    return tb


def add_img_fit(slide, path, bx, by, bw, bh):
    with Image.open(path) as im:
        iw, ih = im.size
    asp = iw / ih; basp = bw / bh
    if asp > basp:
        w = bw; h = bw / asp
    else:
        h = bh; w = bh * asp
    return slide.shapes.add_picture(path, Inches(bx + (bw - w) / 2), Inches(by + (bh - h) / 2), width=Inches(w), height=Inches(h))


def add_chrome(slide, section, title, speaker, num, accent=CYAN):
    add_backdrop(slide, hero=False)
    add_glow_bar(slide, 0, 0, 13.333, 0.06, accent)
    add_txt(slide, 0.65, 0.22, 6.5, 0.28, section.upper(), size=10.5, bold=True, color=accent, font=FONT_MONO)
    add_txt(slide, 0.65, 0.48, 12.0, 0.58, title, size=27, bold=True, color=TXT_MAIN, font=FONT_HEAD)
    add_glow_bar(slide, 0.65, 1.12, 12.03, 0.015, BORDER_DIM)
    # Footer HUD
    add_glow_bar(slide, 0.65, 7.05, 12.03, 0.012, BORDER_DIM)
    add_txt(slide, 0.65, 7.10, 7.0, 0.28,
            "TAXI-V4 RL TELEMETRY  |  Sijan Pandey (11) · Ashim Shrestha (22) · Nishan (TBD)",
            size=10, color=TXT_MUTED, font=FONT_MONO)
    add_txt(slide, 8.2, 7.10, 3.6, 0.28, f"SPEAKER: {speaker.upper()}", size=10, bold=True, color=TXT_SUB, align=PP_ALIGN.RIGHT, font=FONT_MONO)
    add_txt(slide, 11.9, 7.10, 0.78, 0.28, f"{num:02d} / 15", size=10.5, bold=True, color=accent, align=PP_ALIGN.RIGHT, font=FONT_MONO)


def set_notes(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt


# --- SLIDE 1: HERO ---
s1 = prs.slides.add_slide(BLANK)
add_backdrop(s1, hero=True)
add_slide_transition(s1, "fade")
add_glow_bar(s1, 0, 0, 0.16, 7.5, GOLD)

add_pill(s1, 0.78, 0.68, 4.45, 0.38, "REINFORCEMENT LEARNING · GYMNASIUM TAXI-V4", CYAN_DARK, CYAN, size=10.5)
add_txt(s1, 0.78, 1.28, 8.2, 1.75, "Autonomous Taxi\nPickup & Drop-off", size=48, bold=True, color=TXT_MAIN)
add_txt(s1, 0.78, 3.16, 8.0, 0.75,
        "A Controlled Comparative Study of SARSA, DQN, PPO, and Discrete SAC",
        size=19, color=TXT_SUB)

# Algorithm pills
px = 0.78
for lbl, bgc, fgc, w in [
    ("SARSA · TD(0)", GOLD_DARK, GOLD, 1.65),
    ("DQN · Replay", CYAN_DARK, CYAN, 1.60),
    ("PPO · Clipped", VIOLET_DK, VIOLET, 1.65),
    ("SAC · Max-Ent", EMERALD_DK, EMERALD, 1.75),
]:
    add_pill(s1, px, 4.05, w, 0.38, lbl, bgc, fgc, size=11)
    px += w + 0.14

# Team & Roll Numbers Telemetry Card
add_card(s1, 0.78, 4.92, 8.25, 1.92, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.3)
add_txt(s1, 1.05, 5.06, 4.5, 0.28, "PROJECT TEAM & ROLL NUMBERS", size=10.5, bold=True, color=GOLD, font=FONT_MONO)
mx = 1.05
for nm, rn, col in [
    ("Sijan Pandey", "Roll No. 11", CYAN),
    ("Ashim Shrestha", "Roll No. 22", VIOLET),
    ("Nishan", "Roll No. [TBD]", EMERALD),
]:
    add_glow_bar(s1, mx, 5.52, 0.08, 1.04, col)
    add_txt(s1, mx + 0.20, 5.50, 2.35, 0.42, nm, size=17.5, bold=True, color=TXT_MAIN)
    add_txt(s1, mx + 0.20, 6.00, 2.35, 0.38, rn, size=14, bold=True, color=col, font=FONT_MONO)
    mx += 2.65

# Right Telemetry HUD Cards
ky = 1.15
for val, lbl, sub, col, bgc in [
    ("500", "DISCRETE MDP STATES", "5×5 grid · 5 pass · 4 dest", GOLD, GOLD_DARK),
    ("596 eps", "FASTEST CONVERGENCE", "DQN · 37% faster than SARSA", CYAN, CYAN_DARK),
    ("−82%", "PENALTY REDUCTION", "SAC · 2,445 vs 13,551 illegal", EMERALD, EMERALD_DK),
    ("100%", "OPTIMAL DELIVERY", "+7.92 return · 13.08 mean steps", VIOLET, VIOLET_DK),
]:
    add_card(s1, 9.48, ky, 3.22, 1.25, fill=CARD_DARK, border=col, border_pt=1.3)
    add_txt(s1, 9.70, ky + 0.10, 2.80, 0.48, val, size=25, bold=True, color=col, font=FONT_MONO)
    add_txt(s1, 9.70, ky + 0.62, 2.80, 0.26, lbl, size=10, bold=True, color=TXT_MAIN, font=FONT_MONO)
    add_txt(s1, 9.70, ky + 0.88, 2.80, 0.28, sub, size=10.5, color=TXT_MUTED)
    ky += 1.44

set_notes(s1, "[00:00–00:40 | Sijan Pandey, Roll No. 11] Welcome to our Reinforcement Learning mini-project! Sijan (11), Ashim (22), and Nishan present a controlled comparative benchmark of SARSA, DQN, PPO, and Discrete SAC on Gymnasium Taxi-v4.")

# --- SLIDE 2: LIVE HOOK ---
s2 = prs.slides.add_slide(BLANK)
add_chrome(s2, "01 · Live Hook", "All 4 Agents Master Taxi-v4 — Yet Learn Radically Differently", "Sijan Pandey", 2, GOLD)
add_slide_transition(s2, "push")

add_card(s2, 0.65, 1.32, 8.20, 5.50, fill=CARD_DARK, border=GOLD, border_pt=1.3)
add_img_fit(s2, os.path.join(ASSET_DIR, "fig5_taxi_animation_comparison.gif"), 0.78, 1.44, 7.94, 4.92)
add_txt(s2, 0.80, 6.42, 7.90, 0.30,
        "Live 2×2 rollout: Empty (0,1) → Pickup Y(4,0) → Dropoff G(0,4) in 14 optimal steps.",
        size=11.5, color=TXT_MUTED, align=PP_ALIGN.CENTER, font=FONT_MONO)

cy = 1.32
for tag, head, desc, col, bgc in [
    ("SPEED CHAMPION", "DQN · 596 Episodes", "Experience Replay propagates sparse +20 rewards 37% faster than SARSA.", CYAN, CYAN_DARK),
    ("SAFETY CHAMPION", "SAC · −82% Penalties", "Boltzmann max-entropy suppresses −10 actions: 2,445 vs 13,551 for SARSA.", EMERALD, EMERALD_DK),
    ("CENTRAL QUESTION", "Why the Huge Gap?", "Same grid, same optimal 13.08-step policy — radically different learning dynamics.", GOLD, GOLD_DARK),
]:
    add_card(s2, 9.08, cy, 3.60, 1.68, fill=CARD_DARK, border=col, border_pt=1.4)
    add_pill(s2, 9.26, cy + 0.16, 2.05, 0.28, tag, bgc, col, size=9.5)
    add_txt(s2, 9.26, cy + 0.50, 3.24, 0.36, head, size=17, bold=True, color=TXT_MAIN)
    add_txt(s2, 9.26, cy + 0.90, 3.24, 0.68, desc, size=12.2, color=TXT_SUB)
    cy += 1.91
set_notes(s2, "[00:40–01:45 | Sijan] Let the 2x2 GIF play! Point out blue Stage-1 pickup arrows, cyan passenger badge, green Stage-2 delivery arrows. Pose the central question: why did DQN learn 37% faster while SAC cut penalties by 82%?")

# --- SLIDE 3: PROBLEM STATEMENT ---
s3 = prs.slides.add_slide(BLANK)
add_chrome(s3, "02 · Environment", "Gymnasium Taxi-v4: Two-Stage Sequential Decision Task", "Sijan Pandey", 3, CYAN)
add_slide_transition(s3, "fade")

add_card(s3, 0.65, 1.30, 12.03, 3.92, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_img_fit(s3, fig1_dark, 0.78, 1.38, 11.77, 3.76)

sx = 0.65
for head, desc, col in [
    ("Stage 1 · Locate & Pickup", "Navigate around 6 impassable vertical walls to reach waiting station (R, G, Y, B) and trigger Pickup.", CYAN),
    ("Stage 2 · Route & Dropoff", "Once passenger boards (p = 4), re-route through the open central corridor to destination for +20.", EMERALD),
    ("Why It Challenges RL", "Delayed +20 credit arrives after ~13 steps, while blind Pickup/Dropoff hits an instant −10 penalty.", ROSE),
]:
    add_card(s3, sx, 5.40, 3.85, 1.48, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.1)
    add_glow_bar(s3, sx, 5.40, 3.85, 0.06, col)
    add_txt(s3, sx + 0.16, 5.56, 3.53, 0.34, head, size=14.5, bold=True, color=col)
    add_txt(s3, sx + 0.16, 5.92, 3.53, 0.88, desc, size=12, color=TXT_SUB)
    sx += 4.09
set_notes(s3, "[01:45–02:55 | Sijan] Explain the two-stage nature of Dietterich's Taxi-v4: Stage 1 empty pickup around 6 walls, Stage 2 loaded delivery to target station, plus sparse +20 vs harsh -10 penalty.")

# --- SLIDE 4: MDP FORMULATION ---
s4 = prs.slides.add_slide(BLANK)
add_chrome(s4, "02 · MDP Specification", "Formal MDP Formulation: (S, A, P, R, γ)", "Sijan Pandey", 4, CYAN)
add_slide_transition(s4, "push")

add_card(s4, 0.65, 1.35, 5.55, 5.46, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_pill(s4, 0.88, 1.54, 2.50, 0.32, "STATE SPACE · |S| = 500", CYAN_DARK, CYAN, size=10.5)
add_card(s4, 0.88, 2.02, 5.08, 0.92, fill=CARD_ELEV, border=CYAN, border_pt=1.0)
add_img_fit(s4, eq_state, 0.94, 2.08, 4.96, 0.80)
add_bullets(s4, 0.88, 3.16, 5.08, 3.45, [
    ("Grid Coordinates (r, c):", "5 × 5 = 25 discrete city intersections."),
    ("Passenger Status (p):", "5 states — at R, G, Y, B, or 4 (inside taxi)."),
    ("Destination Index (d):", "4 target stations (R, G, Y, B)."),
    ("Deterministic P:", "Walls block East/West moves with zero bounce penalty."),
    ("Discount (γ = 0.95):", "Carries +20 terminal value across the ~13-step horizon."),
], size=14.0, accent=CYAN, space_after=10)

add_card(s4, 6.45, 1.35, 6.23, 5.46, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_pill(s4, 6.68, 1.54, 3.15, 0.32, "ACTIONS · |A| = 6 & REWARDS", EMERALD_DK, EMERALD, size=10.5)

t4 = s4.shapes.add_table(7, 4, Inches(6.68), Inches(2.06), Inches(5.76), Inches(3.85)).table
for i, w in enumerate([0.72, 1.58, 2.02, 1.44]):
    t4.columns[i].width = Inches(w)

for j, h in enumerate(["ID", "Action", "Transition", "Reward"]):
    c = t4.cell(0, j); c.fill.solid(); c.fill.fore_color.rgb = CARD_ELEV; c.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j in (0, 3) else PP_ALIGN.LEFT
    r = p.add_run(); r.text = h; r.font.name = FONT_MONO; r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = CYAN

rows4 = [
    ["0", "Move South (↓)", "(r + 1, c)", "−1"],
    ["1", "Move North (↑)", "(r − 1, c)", "−1"],
    ["2", "Move East (→)", "(r, c + 1) no wall", "−1"],
    ["3", "Move West (←)", "(r, c − 1) no wall", "−1"],
    ["4", "Pickup (P)", "p = 4 at station", "−1 / −10"],
    ["5", "Dropoff (D)", "Goal termination", "+20 / −10"],
]
for i, row in enumerate(rows4, 1):
    bg = CARD_DARK if i % 2 == 1 else CARD_ELEV
    for j, val in enumerate(row):
        c = t4.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = bg; c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j in (0, 3) else PP_ALIGN.LEFT
        r = p.add_run(); r.text = val; r.font.name = FONT_BODY; r.font.size = Pt(12.5); r.font.bold = (j in (1, 3))
        if j == 3 and "+20" in val:
            r.font.color.rgb = EMERALD
        elif j == 3 and "−10" in val:
            r.font.color.rgb = ROSE
        else:
            r.font.color.rgb = TXT_MAIN
add_txt(s4, 6.68, 6.10, 5.76, 0.50,
        "Episode ends on valid Dropoff (+20) or reaches a 200-step cap.",
        size=12.5, bold=True, color=TXT_MUTED)
set_notes(s4, "[02:55–04:10 | Sijan] Walk through state encoding (25 x 5 x 4 = 500 states), 6 actions, and asymmetric rewards (-1 step, -10 illegal, +20 terminal delivery).")

# --- SLIDE 5: CONTROLLED ABLATION ---
s5 = prs.slides.add_slide(BLANK)
add_chrome(s5, "03 · Experimental Design", "Controlled Pure-NumPy Ablation Methodology", "Sijan Pandey", 5, CYAN)
add_slide_transition(s5, "fade")

add_card(s5, 0.65, 1.35, 12.03, 1.50, fill=CARD_ELEV, border=GOLD, border_pt=1.5)
add_pill(s5, 0.90, 1.52, 3.40, 0.30, "WHY CONTROLLED ABLATION?", GOLD_DARK, GOLD, size=10.5)
add_txt(s5, 0.90, 1.92, 11.50, 0.80,
        "All 4 algorithms use exact / one-hot parameterized representations in pure NumPy over the 500-state MDP.\nThis isolates each core mechanism (Replay + Target Sync, Clipping, Max-Entropy) without neural-net noise.",
        size=15.0, color=TXT_MAIN)

px = 0.65
for big, ttl, sub, col in [
    ("2,500", "Training Episodes", "Max 200 steps/ep · fixed seed=42 for exact reproducibility", CYAN),
    ("500", "Greedy Test Episodes", "Evaluated at ε = 0 across unseen start configurations", EMERALD),
    ("4", "Telemetry Metrics", "Return, path length, −10 penalties, episodes to +5.0", VIOLET),
    ("3", "Sensitivity Sweeps", "3-seed controlled SARSA sweeps over α, γ, and ε", GOLD),
]:
    add_card(s5, px, 3.12, 2.84, 2.12, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
    add_glow_bar(s5, px, 3.12, 2.84, 0.06, col)
    add_txt(s5, px + 0.18, 3.32, 2.48, 0.56, big, size=32, bold=True, color=col, font=FONT_MONO)
    add_txt(s5, px + 0.18, 3.96, 2.48, 0.34, ttl, size=14.5, bold=True, color=TXT_MAIN)
    add_txt(s5, px + 0.18, 4.34, 2.48, 0.78, sub, size=11.8, color=TXT_SUB)
    px += 3.06

add_card(s5, 0.65, 5.50, 12.03, 1.32, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.1)
add_txt(s5, 0.90, 5.62, 11.5, 0.28, "UP NEXT — FOUR RL PARADIGMS PRESENTED BY ASHIM SHRESTHA (ROLL NO. 22)", size=10.5, bold=True, color=TXT_MUTED, font=FONT_MONO)
bx = 0.90
for nm, sb, col in [
    ("01 · SARSA", "On-Policy TD(0)", GOLD),
    ("02 · DQN", "Off-Policy Replay", CYAN),
    ("03 · PPO", "Clipped Actor-Critic", VIOLET),
    ("04 · Discrete SAC", "Max-Entropy Twin Q", EMERALD),
]:
    add_glow_bar(s5, bx, 6.00, 0.08, 0.62, col)
    add_txt(s5, bx + 0.20, 5.98, 2.55, 0.34, nm, size=15, bold=True, color=TXT_MAIN)
    add_txt(s5, bx + 0.20, 6.32, 2.55, 0.28, sb, size=12, color=col)
    bx += 2.90
set_notes(s5, "[04:10–05:20 | Sijan -> Handoff to Ashim] Highlight why pure-NumPy one-hot ablation is a strength: clean mechanism isolation without deep-net tuning noise. Hand off smoothly to Ashim!")

# --- SLIDE 6: SARSA ---
s6 = prs.slides.add_slide(BLANK)
add_chrome(s6, "04 · Value-Based RL", "SARSA: On-Policy Temporal-Difference Control", "Ashim Shrestha", 6, GOLD)
add_slide_transition(s6, "push")

add_card(s6, 0.65, 1.35, 12.03, 1.54, fill=CARD_ELEV, border=GOLD, border_pt=1.6)
add_pill(s6, 0.88, 1.48, 2.65, 0.28, "ON-POLICY TD(0) UPDATE", GOLD_DARK, GOLD, size=10)
add_img_fit(s6, eq_sarsa, 0.95, 1.84, 11.4, 0.95)

add_card(s6, 0.65, 3.16, 7.25, 3.66, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_txt(s6, 0.92, 3.34, 6.7, 0.36, "Core Mechanics & Behavior", size=17.5, bold=True, color=GOLD)
add_bullets(s6, 0.92, 3.82, 6.7, 2.80, [
    ("On-Policy Target:", "Bootstraps with the actual next action A' sampled under ε-greedy."),
    ("Single-Step Credit:", "Updates one (s, a) cell per step; +20 propagates backward slowly."),
    ("Penalty Exposure:", "Uniform random actions hit illegal Pickup/Dropoff with prob. ε/3."),
    ("Exact Baseline:", "Zero replay overhead; reaches 100% optimal policy once ε → 0.01."),
], size=14.2, accent=GOLD, space_after=9)

add_card(s6, 8.15, 3.16, 4.53, 3.66, fill=CARD_ELEV, border=BORDER_DIM, border_pt=1.2)
add_txt(s6, 8.42, 3.34, 4.0, 0.36, "SARSA Telemetry", size=17, bold=True, color=TXT_MAIN)
cy = 3.88
for k, v in [
    ("Learning Rate (α)", "0.25"),
    ("Discount Factor (γ)", "0.95"),
    ("Exploration (ε)", "1.0 → 0.01 (×0.996)"),
    ("Episodes to +5.0", "944 episodes"),
    ("Illegal (−10) Actions", "13,551 (Highest)"),
    ("Greedy Test Return", "+7.92 (100% pass)"),
]:
    add_txt(s6, 8.42, cy, 2.15, 0.36, k, size=12.8, color=TXT_MUTED)
    add_txt(s6, 10.45, cy, 2.02, 0.36, v, size=12.8, bold=True,
            color=ROSE if "Highest" in v else (GOLD if "944" in v else TXT_MAIN), align=PP_ALIGN.RIGHT, font=FONT_MONO)
    cy += 0.45
set_notes(s6, "[05:20–06:25 | Ashim Shrestha, Roll No. 22] Explain on-policy SARSA TD(0), why single-step updates take 944 eps and uniform eps-greedy causes 13,551 penalties.")

# --- SLIDE 7: DQN ---
s7 = prs.slides.add_slide(BLANK)
add_chrome(s7, "04 · Value-Based RL", "DQN: Off-Policy Learning with Replay & Target Sync", "Ashim Shrestha", 7, CYAN)
add_slide_transition(s7, "push")

add_card(s7, 0.65, 1.35, 12.03, 1.54, fill=CARD_ELEV, border=CYAN, border_pt=1.6)
add_pill(s7, 0.88, 1.48, 3.10, 0.28, "OFF-POLICY BELLMAN LOSS", CYAN_DARK, CYAN, size=10)
add_img_fit(s7, eq_dqn, 0.90, 1.84, 11.5, 0.95)

px = 0.65
for idx, (head, desc, col) in enumerate([
    ("1. Experience Replay (D)", "Circular buffer stores 12,000 transitions and samples random 32-item mini-batches each step.", CYAN),
    ("2. Target Network (θ⁻)", "Frozen target weights sync every 150 steps to stabilize max Q(s', a') off-policy targets.", VIOLET),
    ("3. Fast Propagation", "Replaying rare +20 dropoffs propagates value across 500 states 37% faster than SARSA!", EMERALD),
]):
    add_card(s7, px, 3.14, 3.72, 1.88, fill=CARD_DARK, border=col, border_pt=1.4)
    add_txt(s7, px + 0.18, 3.28, 3.36, 0.36, head, size=15.5, bold=True, color=col)
    add_txt(s7, px + 0.18, 3.72, 3.36, 1.18, desc, size=13.2, color=TXT_SUB)
    if idx < 2:
        add_txt(s7, px + 3.74, 3.82, 0.40, 0.48, "→", size=24, bold=True, color=TXT_MUTED, align=PP_ALIGN.CENTER)
    px += 4.15

add_card(s7, 0.65, 5.26, 12.03, 1.54, fill=CARD_ELEV, border=BORDER_DIM, border_pt=1.2)
add_txt(s7, 0.90, 5.38, 6.0, 0.30, "BENCHMARK VERDICT · SAMPLE EFFICIENCY CHAMPION", size=10.8, bold=True, color=CYAN, font=FONT_MONO)
add_bullets(s7, 0.90, 5.74, 11.45, 0.98, [
    ("Fastest Convergence (596 episodes):", "Breaks sequential correlation and extracts maximum signal from sparse +20 dropoffs."),
    ("Remaining Trade-off (10,879 penalties):", "Uniform ε-greedy exploration still triggers random −10 penalties early in training."),
], size=14.0, accent=CYAN, space_after=6)
set_notes(s7, "[06:25–07:35 | Ashim] Walk through Replay Buffer (12,000), 32-batch sampling, Target Sync (150 steps). Emphasize 596 episodes (37% faster) vs remaining eps-greedy penalty weakness.")

# --- SLIDE 8: PPO ---
s8 = prs.slides.add_slide(BLANK)
add_chrome(s8, "05 · Policy Gradient RL", "PPO: On-Policy Clipped Actor-Critic", "Ashim Shrestha", 8, VIOLET)
add_slide_transition(s8, "push")

add_card(s8, 0.65, 1.35, 12.03, 1.54, fill=CARD_ELEV, border=VIOLET, border_pt=1.6)
add_pill(s8, 0.88, 1.48, 3.25, 0.28, "CLIPPED SURROGATE OBJECTIVE", VIOLET_DK, VIOLET, size=10)
add_img_fit(s8, eq_ppo, 0.88, 1.84, 11.55, 0.95)

add_card(s8, 0.65, 3.14, 6.45, 3.68, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_txt(s8, 0.90, 3.30, 5.95, 0.36, "How Clipping Protects the Policy", size=17, bold=True, color=VIOLET)
add_bullets(s8, 0.90, 3.76, 5.95, 2.90, [
    ("Softmax Actor + Critic:", "Directly optimizes π(a|s) with advantage A = Q(s,a) − V(s)."),
    ("Trust Region (ε = 0.20):", "Clips ratio r(θ) to [0.8, 1.2], preventing policy collapse."),
    ("53% Safer Than SARSA:", "Softmax suppresses −10 moves (6,356 vs 13,551 penalties)."),
    ("On-Policy Trade-off:", "Discards past episodes → 1,023 eps to +5.0; 99.8% success."),
], size=13.8, accent=VIOLET, space_after=9)

add_card(s8, 7.32, 3.14, 5.36, 3.68, fill=CARD_ELEV, border=VIOLET, border_pt=1.2)
add_img_fit(s8, ppo_clip_dark, 7.42, 3.24, 5.16, 3.48)
set_notes(s8, "[07:35–08:40 | Ashim] Use the PPO clipped surrogate curve on the right to explain how bounding r(theta) to [0.8, 1.2] stops destructive policy updates while cutting penalties 53% vs SARSA.")

# --- SLIDE 9: DISCRETE SAC ---
s9 = prs.slides.add_slide(BLANK)
add_chrome(s9, "05 · Maximum-Entropy RL", "Discrete SAC: Twin Q-Critics & Boltzmann Safety", "Ashim Shrestha", 9, EMERALD)
add_slide_transition(s9, "push")

add_card(s9, 0.65, 1.35, 6.20, 2.52, fill=CARD_ELEV, border=EMERALD, border_pt=1.6)
add_pill(s9, 0.86, 1.47, 3.65, 0.28, "SOFT BELLMAN & BOLTZMANN POLICY", EMERALD_DK, EMERALD, size=10)
add_img_fit(s9, eq_sac_v, 0.80, 1.80, 5.90, 0.88)
add_img_fit(s9, eq_sac_pi, 0.80, 2.68, 5.90, 1.08)

add_card(s9, 0.65, 4.08, 6.20, 2.74, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_bullets(s9, 0.88, 4.24, 5.75, 2.45, [
    ("Twin Q-Critics (Q₁, Q₂):", "min(Q₁, Q₂) removes Q-value overestimation bias."),
    ("Annealed Entropy (τ: 1.2 → 0.08):", "Maximizes reward + entropy for smart exploration."),
    ("−82% Fewer Penalties (2,445):", "One −10 penalty makes exp(−10/τ) ≈ 0 instantly!"),
], size=13.8, accent=EMERALD, space_after=10)

add_card(s9, 7.08, 1.35, 5.60, 5.47, fill=CARD_ELEV, border=EMERALD, border_pt=1.3)
add_txt(s9, 7.30, 1.50, 5.15, 0.38, "Why SAC Slashes Catastrophic Penalties", size=16.2, bold=True, color=TXT_MAIN)
add_img_fit(s9, sac_bars_dark, 7.18, 1.96, 5.40, 3.62)
add_card(s9, 7.30, 5.74, 5.16, 0.88, fill=EMERALD_DK, border=EMERALD, border_pt=1.1)
add_txt(s9, 7.46, 5.84, 4.86, 0.68,
        "Insight: ε-greedy wastes 10% of actions on known −10 errors; SAC explores ONLY among competitive valid moves.",
        size=12.2, bold=True, color=EMERALD)
set_notes(s9, "[08:40–09:50 | Ashim] Walk through the Boltzmann vs eps-greedy chart! Explain why exp(-10/tau) kills illegal action probability to ~0.002%, producing an 82% penalty reduction.")

# --- SLIDE 10: 4 PARADIGMS AT A GLANCE ---
s10 = prs.slides.add_slide(BLANK)
add_chrome(s10, "05 · Synthesis", "Four Modern RL Paradigms at a Glance", "Ashim Shrestha", 10, VIOLET)
add_slide_transition(s10, "fade")

cx = 0.65
for name, tag, col, bgc, rows in [
    ("SARSA", "On-Policy TD(0)", GOLD, GOLD_DARK, [
        ("Update Style", "Single-step online TD"), ("Policy Type", "Value-based Q-table"),
        ("Exploration", "Uniform ε-greedy"), ("Core Strength", "Simple, exact baseline"), ("Trade-off", "Slow + highest −10s"),
    ]),
    ("DQN", "Off-Policy Value", CYAN, CYAN_DARK, [
        ("Update Style", "Replay Buffer (32 batch)"), ("Policy Type", "Q-net + Target Sync"),
        ("Exploration", "Uniform ε-greedy"), ("Core Strength", "Fastest (596 episodes)"), ("Trade-off", "Random −10 penalties"),
    ]),
    ("PPO", "On-Policy Clipped AC", VIOLET, VIOLET_DK, [
        ("Update Style", "Multi-epoch Clipped PG"), ("Policy Type", "Softmax Actor + Critic"),
        ("Exploration", "Learned Softmax + H(π)"), ("Core Strength", "Smooth, stable updates"), ("Trade-off", "Discards old rollouts"),
    ]),
    ("Discrete SAC", "Off-Policy Max-Ent", EMERALD, EMERALD_DK, [
        ("Update Style", "Soft Bellman + Twin Q"), ("Policy Type", "Boltzmann Energy Policy"),
        ("Exploration", "Temperature-scaled (τ)"), ("Core Strength", "Safest (−82% penalties)"), ("Trade-off", "Needs τ & twin critics"),
    ]),
]:
    add_card(s10, cx, 1.38, 2.86, 5.44, fill=CARD_DARK, border=col, border_pt=1.5)
    add_card(s10, cx, 1.38, 2.86, 0.90, fill=bgc, border=col, border_pt=1.2, rounded=False)
    add_txt(s10, cx + 0.12, 1.48, 2.62, 0.38, name, size=19.5, bold=True, color=col, align=PP_ALIGN.CENTER)
    add_txt(s10, cx + 0.12, 1.88, 2.62, 0.28, tag, size=11.5, bold=True, color=TXT_MAIN, align=PP_ALIGN.CENTER)
    ry = 2.48
    for k, v in rows:
        add_txt(s10, cx + 0.18, ry, 2.50, 0.24, k.upper(), size=9.2, bold=True, color=TXT_MUTED, font=FONT_MONO)
        add_txt(s10, cx + 0.18, ry + 0.23, 2.50, 0.40, v, size=13.2, bold=(k == "Core Strength"),
                color=col if k == "Core Strength" else TXT_MAIN)
        ry += 0.85
    cx += 3.06
set_notes(s10, "[09:50–10:40 | Ashim -> Handoff to Nishan] Summarize all 4 paradigms across the on/off-policy and value/actor-critic axes. Hand off to Nishan for experimental results!")

# --- SLIDE 11: LEARNING CURVES ---
s11 = prs.slides.add_slide(BLANK)
add_chrome(s11, "06 · Benchmark Curves", "Head-to-Head Learning Dynamics (2,500 Episodes)", "Nishan", 11, EMERALD)
add_slide_transition(s11, "push")

add_card(s11, 0.65, 1.30, 8.95, 5.54, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_img_fit(s11, fig2_dark, 0.74, 1.38, 8.77, 5.38)

cy = 1.30
for head, desc, col in [
    ("Panels A–B · Return & Steps", "DQN (cyan) surges to optimal +7.9 return and 13.1 steps far ahead of competitors.", CYAN),
    ("Panel C · Illegal (−10) Safety", "SAC (emerald) starts lowest and plunges to zero almost immediately.", EMERALD),
    ("Panel D · Episodes to +5.0", "Off-policy dominates: DQN (596) & SAC (808) beat SARSA (944) & PPO (1,023).", VIOLET),
]:
    add_card(s11, 9.82, cy, 2.86, 1.71, fill=CARD_DARK, border=col, border_pt=1.4)
    add_glow_bar(s11, 9.82, cy, 2.86, 0.06, col)
    add_txt(s11, 9.96, cy + 0.18, 2.60, 0.34, head, size=12.8, bold=True, color=col)
    add_txt(s11, 9.96, cy + 0.58, 2.60, 1.00, desc, size=12.2, color=TXT_SUB)
    cy += 1.91
set_notes(s11, "[10:40–12:00 | Nishan] Walk panel-by-panel through fig2: Panel A/B show DQN fastest climb; Panel C shows SAC near-zero penalties; Panel D compares episodes to +5.0.")

# --- SLIDE 12: QUANTITATIVE LEADERBOARD ---
s12 = prs.slides.add_slide(BLANK)
add_chrome(s12, "06 · Quantitative Leaderboard", "Evaluation Summary: 2,500 Train / 500 Greedy Test Episodes", "Nishan", 12, EMERALD)
add_slide_transition(s12, "fade")

t12 = s12.shapes.add_table(5, 6, Inches(0.65), Inches(1.42), Inches(12.03), Inches(3.38)).table
for i, w in enumerate([1.55, 2.55, 2.15, 2.05, 1.95, 1.78]):
    t12.columns[i].width = Inches(w)

for j, h in enumerate(["Algorithm", "Paradigm", "Episodes to +5.0", "Total −10 Penalties", "Greedy Return (ε=0)", "Success / Steps"]):
    c = t12.cell(0, j); c.fill.solid(); c.fill.fore_color.rgb = CARD_ELEV; c.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT
    r = p.add_run(); r.text = h; r.font.name = FONT_MONO; r.font.size = Pt(11.8); r.font.bold = True; r.font.color.rgb = CYAN

rows12 = [
    ["DQN", "Off-Policy + Replay", "596 eps  ★ Fastest", "10,879", "+7.92 ± 2.58", "100.0% / 13.08"],
    ["Discrete SAC", "Off-Policy Max-Entropy", "808 eps", "2,445  ★ Safest", "+7.92 ± 2.58", "100.0% / 13.08"],
    ["SARSA", "On-Policy TD(0)", "944 eps", "13,551", "+7.92 ± 2.58", "100.0% / 13.08"],
    ["PPO", "On-Policy Clipped AC", "1,023 eps", "6,356", "+7.88 ± 2.61", "99.8% / 13.12"],
]
acols = [CYAN, EMERALD, GOLD, VIOLET]
for i, row in enumerate(rows12, 1):
    bg = CARD_DARK if i % 2 == 1 else CARD_ELEV
    for j, val in enumerate(row):
        c = t12.cell(i, j); c.fill.solid()
        if "Fastest" in val:
            c.fill.fore_color.rgb = CYAN_DARK
        elif "Safest" in val:
            c.fill.fore_color.rgb = EMERALD_DK
        else:
            c.fill.fore_color.rgb = bg
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT
        r = p.add_run(); r.text = val; r.font.name = FONT_BODY; r.font.size = Pt(13.2); r.font.bold = (j == 0 or "★" in val)
        if j == 0:
            r.font.color.rgb = acols[i - 1]
        elif "Fastest" in val:
            r.font.color.rgb = CYAN
        elif "Safest" in val:
            r.font.color.rgb = EMERALD
        else:
            r.font.color.rgb = TXT_MAIN

add_card(s12, 0.65, 5.10, 5.85, 1.72, fill=CYAN_DARK, border=CYAN, border_pt=1.5)
add_txt(s12, 0.90, 5.24, 5.4, 0.38, "★ Sample Efficiency Champion → DQN", size=16.5, bold=True, color=CYAN)
add_txt(s12, 0.90, 5.68, 5.4, 0.95,
        "Reaches +5.0 return in 596 episodes — 37% faster than SARSA (944) and 42% faster than PPO (1,023) via replay.",
        size=13.2, color=TXT_MAIN)

add_card(s12, 6.83, 5.10, 5.85, 1.72, fill=EMERALD_DK, border=EMERALD, border_pt=1.5)
add_txt(s12, 7.08, 5.24, 5.4, 0.38, "★ Exploration Safety Champion → Discrete SAC", size=16.5, bold=True, color=EMERALD)
add_txt(s12, 7.08, 5.68, 5.4, 0.95,
        "Only 2,445 illegal penalties — 82% fewer than SARSA (13,551) and 78% fewer than DQN (10,879), at 100% delivery.",
        size=13.2, color=TXT_MAIN)
set_notes(s12, "[12:00–13:00 | Nishan] Highlight the two category winners (DQN fastest at 596 eps; SAC safest at 2,445 penalties) and note all four reach 99.8–100% test delivery.")

# --- SLIDE 13: HYPERPARAMETER SENSITIVITY ---
s13 = prs.slides.add_slide(BLANK)
add_chrome(s13, "07 · Sensitivity Sweeps", "Hyperparameter Sensitivity Across α, γ, and ε", "Nishan", 13, GOLD)
add_slide_transition(s13, "push")

add_card(s13, 0.65, 1.30, 12.03, 3.84, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_img_fit(s13, fig3_dark, 0.78, 1.38, 11.77, 3.68)

hx = 0.65
for head, best, desc, col in [
    ("A. Learning Rate (α)", "Best: α ∈ [0.20, 0.50]", "α = 0.05 is sluggish; aggressive α = 0.90 oscillates near −100 by overshooting noisy ε-greedy TD targets.", CYAN),
    ("B. Discount Factor (γ)", "Best: γ ∈ [0.95, 0.99]", "Myopic γ = 0.50 decays +20 across 13 steps (0.5¹³ ≈ 0.0001), breaking credit assignment to pickup.", EMERALD),
    ("C. Exploration Rate (ε)", "Best: Decaying ε (1.0 → 0.01)", "Fixed ε = 0.30 stalls below −25 from constant −10 penalties; decaying ε balances discovery + exploitation.", VIOLET),
]:
    add_card(s13, hx, 5.30, 3.85, 1.56, fill=CARD_DARK, border=col, border_pt=1.3)
    add_txt(s13, hx + 0.16, 5.40, 3.53, 0.30, head, size=14, bold=True, color=TXT_MAIN)
    add_txt(s13, hx + 0.16, 5.71, 3.53, 0.26, best, size=11.8, bold=True, color=col, font=FONT_MONO)
    add_txt(s13, hx + 0.16, 6.00, 3.53, 0.80, desc, size=11.6, color=TXT_SUB)
    hx += 4.09
set_notes(s13, "[13:00–14:05 | Nishan] Explain the three clear failure modes: alpha=0.90 overshooting at -100, gamma=0.50 killing long-horizon credit (0.5^13), and fixed eps=0.30 capping return below -25.")

# --- SLIDE 14: HEATMAPS & EXPLAINABILITY ---
s14 = prs.slides.add_slide(BLANK)
add_chrome(s14, "08 · Explainability", "Inside the Agent's Brain: V*(s) Heatmaps & Optimal Rollout", "Nishan", 14, EMERALD)
add_slide_transition(s14, "push")

add_card(s14, 0.65, 1.30, 12.03, 3.90, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_img_fit(s14, fig4_dark, 0.78, 1.38, 11.77, 3.74)

hx = 0.65
for head, desc, col in [
    ("Stage 1 · Gradient Climbs to Y", "V*(s) rises smoothly toward +5.2 at Station Y(4,0). Every arrow cleanly navigates around brick walls into PICKUP.", CYAN),
    ("Stage 2 · Gradient Flips to G!", "The instant p = 4 (passenger inside), values invert—climbing to +20.0 at G(0,4). Dead-end (4,1) stays unvisited.", GOLD),
    ("Optimal 14-Step Delivery", "Starting from (0,1), the agent completes (0,1) → Y → G in the exact theoretical minimum with zero penalties.", EMERALD),
]:
    add_card(s14, hx, 5.36, 3.85, 1.50, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
    add_glow_bar(s14, hx, 5.36, 3.85, 0.06, col)
    add_txt(s14, hx + 0.16, 5.50, 3.53, 0.32, head, size=13.8, bold=True, color=col)
    add_txt(s14, hx + 0.16, 5.86, 3.53, 0.92, desc, size=11.8, color=TXT_SUB)
    hx += 4.09
set_notes(s14, "[14:05–15:05 | Nishan] Show how V*(s) flips direction between Stage 1 (toward Y) and Stage 2 (toward G), and point out why dead-end cells like (4,1) stay near zero.")

# --- SLIDE 15: CONCLUSIONS & QR ---
s15 = prs.slides.add_slide(BLANK)
add_backdrop(s15, hero=True)
add_slide_transition(s15, "fade")
add_glow_bar(s15, 0, 0, 13.333, 0.08, EMERALD)

add_txt(s15, 0.70, 0.36, 6.0, 0.30, "09 · PRACTICAL DECISION GUIDE & CLOSING", size=11, bold=True, color=GOLD, font=FONT_MONO)
add_txt(s15, 0.70, 0.70, 11.8, 0.62, "Which RL Algorithm Should You Choose?", size=32, bold=True, color=TXT_MAIN)

dx = 0.70
for q, ans, desc, col, bgc in [
    ("Need Fastest Learning?", "Choose DQN", "Off-policy Experience Replay + Target Sync cuts convergence episodes by 37% (596 vs 944).", CYAN, CYAN_DARK),
    ("Need Safe Exploration?", "Choose Discrete SAC", "Boltzmann Max-Entropy + Twin Critics slashes catastrophic −10 penalties by 82% (2,445 vs 13,551).", EMERALD, EMERALD_DK),
    ("Need Policy Stability?", "Choose PPO / SARSA", "PPO guards against policy collapse (99.8%); SARSA gives a zero-overhead 100% baseline.", VIOLET, VIOLET_DK),
]:
    add_card(s15, dx, 1.58, 3.78, 2.40, fill=CARD_DARK, border=col, border_pt=1.6)
    add_glow_bar(s15, dx, 1.58, 3.78, 0.07, col)
    add_txt(s15, dx + 0.22, 1.80, 3.34, 0.32, q, size=12.8, bold=True, color=TXT_SUB)
    add_txt(s15, dx + 0.22, 2.16, 3.34, 0.44, ans, size=22, bold=True, color=col)
    add_txt(s15, dx + 0.22, 2.70, 3.34, 1.15, desc, size=13.2, color=TXT_MAIN)
    dx += 4.07

add_card(s15, 0.70, 4.28, 8.20, 2.58, fill=CARD_DARK, border=BORDER_DIM, border_pt=1.2)
add_txt(s15, 0.96, 4.44, 4.5, 0.30, "FUTURE EXTENSIONS", size=10.5, bold=True, color=GOLD, font=FONT_MONO)
add_txt(s15, 0.96, 4.80, 7.70, 0.80,
        "• Scale to continuous city grids with Deep MLPs  ·  Add stochastic traffic delays & multi-passenger dispatch",
        size=13.5, color=TXT_SUB)
add_glow_bar(s15, 0.96, 5.70, 7.65, 0.015, BORDER_DIM)
add_txt(s15, 0.96, 5.82, 7.65, 0.28, "THANK YOU FOR WATCHING  —  TEAM MEMBERS", size=10.5, bold=True, color=CYAN, font=FONT_MONO)
add_txt(s15, 0.96, 6.18, 7.75, 0.42,
        "Sijan Pandey (Roll No. 11)  |  Ashim Shrestha (Roll No. 22)  |  Nishan (Roll No. [TBD])",
        size=13.5, bold=True, color=TXT_MAIN)

add_card(s15, 9.18, 4.28, 3.45, 2.58, fill=TXT_MAIN, border=CYAN, border_pt=1.5)
add_img_fit(s15, qr_dark, 9.34, 4.42, 1.75, 1.75)
add_txt(s15, 11.16, 4.52, 1.36, 1.50, "Scan for\nGitHub\nCode &\nNotebook", size=13.5, bold=True, color=BG_DARK)
add_txt(s15, 9.28, 6.28, 3.25, 0.44, "github.com/devsijanpandey12/\nRL_FINAL_MINIPROJECT",
        size=9.8, bold=True, color=RGBColor(0x03, 0x69, 0xA1), font=FONT_MONO, align=PP_ALIGN.CENTER)

set_notes(s15, "[15:05–16:00 | Nishan closing for Team] Deliver the 3-way decision guide (DQN for speed, SAC for safety, PPO/SARSA for stability/simplicity), mention future work, point to the GitHub QR code, and thank everyone on behalf of Sijan (11), Ashim (22), and Nishan!")

prs.save(OUT_PPTX)
print(f"[4/5] Saved Next-Gen PPTX: {OUT_PPTX}")
print("[5/5] All dark-mode assets & exact policy JSON generated successfully!")
