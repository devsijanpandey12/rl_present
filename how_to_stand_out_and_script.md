# Taxi-v4 Mission Control — 16-Minute Presentation Script & Stand-Out Playbook

**Deck:** [`index.html`](index.html) — **20-Slide Dark "Midnight Telemetry" Interactive Web Presentation**  
**Theme:** Deep obsidian-slate canvas, animated autonomous taxi traffic on the cover, cinematic **Chapter Interstitials**, animated **Agent–Environment RL loop**, glowing dark-mode equations/plots, and **6 live interactive labs** powered by your real 500-state `Taxi-v4` optimal policy and real 2,500-episode training curves!  
**GitHub Repo:** `https://github.com/devsijanpandey12/RL_FINAL_MINIPROJECT`

> **Keyboard Shortcuts (Chrome / Edge):** `→` / `Space` Next · `←` Back · `F` Fullscreen · `N` Synced Presenter Notes · `T` Start/Pause 16:00 Timer · `G` 20-Slide Grid *(cursor & dock auto-hide after 2.5s idle for clean OBS recording!)*

---

## 3-Person Chapter Split (~16:00 Total · Flexible Slide Assignment)

| Chapter | Suggested Presenter | Roll No. | Slides | Time Window | Creative Showstopper Moments |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Ch. 01** | **Sijan Pandey** | **11** | **01 – 08** | `00:00 – 05:30` | Cover CLI, 4-agent GIF hook, live State-ID Grid, **animated Agent–Environment loop**, **5-ingredient MDP cards**, 3-line NumPy code reveal |
| **Ch. 02** | **Ashim Shrestha** | **22** | **09 – 14** | `05:30 – 10:45` | Replay Propagator demo, PPO Clip Curve, **Live SAC `τ` Lab**, **2×2 RL Quadrant Map**, **Live Taxi-v4 Simulator + `V*(s)` Flip** |
| **Ch. 03** | **Nishan** | **5** | **15 – 20** | `10:45 – 16:00` | **Animated 2,500-Ep Learning Race** (real notebook curves!), Leaderboard bars, Failure modes, `V*(s)` heatmaps, **Interactive Hiring Router** |

> **Metrics Alignment Note:** Every metric in `index.html` and this script matches your exact executed Jupyter notebook (`Taxi_Pickup_Dropoff_RL_Comparison.ipynb`, Cell 13, `seed=42`): **DQN `596` eps / `7,192` penalties / `+7.90` (100%)**, **SAC `808` eps / `1,486` penalties (`−86%`) / `+5.83` (99.0%)**, **SARSA `944` eps / `10,908` penalties / `+7.01` (100%)**, **PPO `1,023` eps / `3,606` penalties (`−67%`) / `+6.60` (99.6%)**.

---

## CHAPTER 01 — SIJAN PANDEY (Roll No. 11) | Slides 01–08 (`00:00 – 05:30`)
*(Written for natural, conversational delivery at ~140 words/min — ~790 words total. Speak like you're explaining to a smart friend, not reading a textbook!)*

### Slide 01 · Mission Briefing & Cover (`00:00 – 00:40` · ~90 words)
**[VISUAL: Let the cover taxis cruise and the CLI line finish typing for 2 seconds as you smile & open]**
> "Hey everyone! Imagine dropping a self-driving taxi into a small city with no GPS, no map rules, and no instructions—it literally has to figure out how to pick up passengers and drop them off purely through trial and error.
>
> That's what our Reinforcement Learning mini-project is all about. I'm **Sijan Pandey**, Roll Number 11, and I built this together with **Ashim Shrestha**, Roll Number 22, and **Nishan**, Roll Number 5.
>
> Over the next 16 minutes, I'll walk you through the problem and how we set up a fair experiment, Ashim will break down four different RL algorithms with live demos, and Nishan will show who actually won and why."

---

### Slide 02 · The Hook: 4-Agent Synchronized Rollout (`00:40 – 01:30` · ~115 words)
**[PAUSE 2 SECONDS — let viewers' eyes lock onto the four moving taxis before speaking]**
> "Before we show you a single formula, just watch these four trained taxis drive the exact same trip side by side.
>
> All four start empty at the top-left, at cell `(0,1)`. Follow the **blue arrows**—each taxi weaves around those vertical walls to reach the waiting passenger down at **Station Yellow**, picks them up—that's when the badge flips cyan—and then follows the **green arrows** across the middle row to drop them off at **Station Green** for a `+20` reward.
>
> Notice something cool: by the end of training, all four algorithms drive the exact same perfect 14-step route. So if they all end up just as smart, why even compare four algorithms? Because *how* they learned it couldn't be more different. One learned **37% faster**, and another made **86% fewer mistakes**. Let's see how the city itself works."

---

### Slide 03 · Interactive `Taxi-v4` City Grid (`01:30 – 02:15` · ~110 words)
**[LIVE DEMO: Click cell `(4,0)`, then click `In taxi` under Passenger so the viewer watches the State ID jump live!]**
> "We used Gymnasium's classic **Taxi-v4** environment, and it's trickier than a normal maze because every episode has two completely different goals stitched together.
>
> Look at our interactive map. When I click a cell—say `(4,0)`—you see a single state ID update on the right. That's because the taxi doesn't just need to know *where it is*; it needs to know three things at once: its grid cell, where the passenger is, and where they want to go.
>
> Watch what happens the moment I click **`In taxi`**: the state ID jumps, and suddenly every optimal arrow on the map has to reverse direction. Throw in six impassable orange walls—plus a painful **`−10` penalty** anytime the taxi presses Pickup or Dropoff in the wrong spot—and random guessing gets punished fast."

---

### Slide 04 · Agent–Environment RL Loop (`02:15 – 02:45` · ~80 words)
**[GESTURE LEFT → RIGHT → LEFT following the animated glowing arrows]**
> "So how does a taxi learn all of this without us hard-coding a single `if-else` rule? Through this simple feedback loop.
>
> At every step, our **Agent** looks at the current situation, checks its policy, and picks an **action `aₜ`**—like moving South or pressing Pickup. The **Taxi-v4 Environment** moves the car against the grid and walls, and answers back with two things: the **next state `sₜ₊₁`** and a **reward `rₜ₊₁`**.
>
> That loop repeats until a successful drop-off finishes the trip—or 200 steps run out—and the agent's only job is to maximize its total discounted reward."

---

### Slide 05 · Five Ingredients of the MDP: `(S, A, P, R, γ)` (`02:45 – 03:15` · ~85 words)
**[SWEEP ACROSS THE 5 CARDS LEFT TO RIGHT: S → A → P → R → γ]**
> "In reinforcement learning, we formalize this loop as a **Markov Decision Process**, or MDP, which sounds intimidating—but it's really just five simple ingredients:
>
> **`S`** is the **500** possible situations the taxi can be in. **`A`** is the **6** buttons it can press. **`P`** is the physics—here it's **deterministic**, meaning if you press North and there's no wall, you always go North. **`R`** is the reward signal: `−1`, `−10`, or `+20`. And **`γ`** is **0.95**, controlling how much the taxi cares about future rewards.
>
> And the **Markov property** at the bottom just means the taxi only needs to know what's happening *right now*—it never needs to remember how it got there."

---

### Slide 06 · Formal MDP Details: State Encoding & Action Table (`03:15 – 04:00` · ~110 words)
**[POINT TO FORMULA ON LEFT, THEN REWARD COLUMN ON RIGHT]**
> "Putting real numbers on those five ingredients: where does **500 states** come from? Simple multiplication—`25` grid squares, times `5` passenger statuses (four waiting stations plus 'inside the taxi'), times `4` possible destinations. That formula at the top packs all four variables into one clean integer from `0` to `499`.
>
> Look at `γ = 0.95`. Since a typical trip takes about 13 steps, `0.95` to the 13th power is still `0.51`—meaning even from the very start of the episode, the taxi can still 'feel' half of that final `+20` reward.
>
> On the right are the 6 actions: every move costs **`−1`** so the taxi hates wasting time, a valid Dropoff pays **`+20`**, and any wrong Pickup or Dropoff slaps it with **`−10`**."

---

### Slide 07 · Controlled Pure-NumPy Ablation (`04:00 – 04:45` · ~110 words)
**[SPEAK CONFIDENTLY — THIS IS A PROFESSOR-IMPRESSING MOMENT]**
> "Now here's a question you might be thinking: why did we build SARSA, DQN, PPO, and SAC from scratch in **pure NumPy** instead of plugging in a huge PyTorch neural network?
>
> Think of it like testing four racecar drivers. If you put each driver in a different car with different tires, you'll never know if the driver won or the car won. Since Taxi-v4 only has 500 exact states, giving all four algorithms the exact same clean NumPy representation creates a **controlled experiment**—so any difference we see comes *purely* from the algorithm's core idea: experience replay, clipping, or entropy.
>
> To keep it completely fair, every agent trains for **2,500 episodes** on the same random seed, gets tested on **500 unseen trips** with exploration turned off, and we also ran **36 extra sweeps** to see how hyperparameters break."

---

### Slide 08 · Inside the Code (From Our GitHub Notebook) (`04:45 – 05:30` · ~105 words)
**[POINT BRIEFLY TO EACH OF THE 4 CODE BOXES — EXPLAIN THE IDEA, DON'T READ SYNTAX!]**
> "Before I hand it over to Ashim, I love this slide because it shows the *actual* core code straight from our GitHub notebook—and you can literally see each algorithm's personality in just three lines of Python.
>
> **SARSA** in yellow updates right away using whatever next move it actually picked. **DQN** in blue grabs a random mini-batch of 32 past memories and updates against a frozen target table synced every 150 steps. **PPO** in purple checks if the policy jumped outside `[0.8, 1.2]`, and if it did, it simply refuses the update. And **Discrete SAC** in green keeps two Q-tables, picks the more cautious minimum, and adds an entropy bonus.
>
> Now let's dive into how each of these four agents actually behaves—**Ashim**, over to you!"

---

## CHAPTER 02 — ASHIM SHRESTHA (Roll No. 22) | Slides 09–14 (`05:30 – 10:45`)
*(Cinematic Chapter 02 interstitial triggers automatically on advancing to Slide 09!)*

### Slide 09 · Agent 01: SARSA (`05:30 – 06:20`)
> "Thanks, Sijan! I'm **Ashim Shrestha**, Roll Number **22**.
>
> Our first agent is **SARSA**—on-policy TD(0). Look at the golden update rule: SARSA bootstraps on the **actual next exploratory action `A_{t+1}`**. Because it updates strictly one cell per step, the `+20` reward creeps backward slowly, taking **944 episodes** to reach `+5.0`. And because uniform `ε`-greedy hits illegal Pickup/Dropoff roughly `ε/3` of the time early on, SARSA racks up **10,908 illegal penalties**—the worst in our benchmark—before converging to **100% greedy delivery (`+7.01`)**."

### Slide 10 · Agent 02: DQN + Interactive Replay Propagator (`06:20 – 07:20`)
**[LIVE CUE: Click `+ Run update cycle` 3 times while narrating!]**
> "Second is **DQN**, which shifts off-policy with a **12,000-transition Experience Replay Buffer** sampled in 32-item mini-batches and a **Target Network** synced every 150 steps.
>
> Watch our interactive demo on the right as I run three update cycles: SARSA advances the `+20` reward just one tile backward per cycle, while **DQN replays the stored corridor and broadcasts value twice as fast**.
>
> That single upgrade cuts convergence down to **596 episodes—37% faster than SARSA**—and earns the highest greedy test return (`+7.90`, `100%`). Its weakness? Still `ε`-greedy, so it incurs **7,192 illegal moves**."

### Slide 11 · Agent 03: PPO + Interactive Trust-Region Curve (`07:20 – 08:20`)
**[LIVE CUE: Toggle `A > 0` → `A < 0` and drag the `clip ε` slider!]**
> "Third is **PPO**—an on-policy Actor-Critic with a Softmax Actor (`lr = 0.35`) and Value Critic (`lr = 0.30`).
>
> Watch the interactive trust-region curve on the right: whether advantage is positive or negative, clipping the probability ratio inside **`[0.80, 1.20]`** flattens the objective so one noisy episode can never wreck a good routing policy.
>
> Because its Softmax actor quickly suppresses `−10` actions, PPO cuts penalties by **67% vs. SARSA** (down to **3,606**) with **99.6% delivery**—though discarding old rollouts pushes convergence to **1,023 episodes**."

### Slide 12 · Agent 04: Discrete SAC + Live `τ` Lab (`08:20 – 09:20`)
**[SHOWSTOPPER LIVE CUE: Slowly drag the `τ` slider from `1.20` down to `0.20`!]**
> "Fourth is **Discrete SAC**, our safety champion. First, **Twin Critics (`min(Q₁, Q₂)`)** prevent overestimation. Second, SAC explores via a **Boltzmann entropy policy** scaled by temperature `τ` annealing from `1.2 → 0.08`.
>
> Watch the live probability bars as I drag `τ`: once an illegal Pickup or Dropoff scores `−10`, **`ε`-greedy on the left STILL hits those red illegal actions 10% of the time**. On the right, **SAC's `exp(−10/τ)` collapses those red bars to ~0.00% immediately**, keeping exploration strictly on sensible grid moves. Result: only **1,486 illegal penalties—an 86% reduction vs. SARSA**!"

### Slide 13 · The RL Map (2×2 Quadrant Synthesis) (`09:20 – 09:55`)
> "Stepping back, our four agents cover all four quadrants of reinforcement learning—on-policy vs. off-policy horizontally, value-based vs. actor-critic vertically.
>
> Two crisp laws emerge visually: **the right column (off-policy DQN `596`, SAC `808`) learns faster**, while **the bottom row (probability-based SAC `1,486`, PPO `3,606`) explores dramatically safer**."

### Slide 14 · LIVE Autonomous `Taxi-v4` Simulator & Handoff (`09:55 – 10:45`)
**[SHOWSTOPPER LIVE CUE: Click `Illegal move −10` once, click `↺`, then hit `▶ Auto-drive` with `V*(s)` ON!]**
> "Before Nishan shows our training race, let's drive the exact 500-state optimal policy live!
>
> Tap **`Illegal move −10`** once—instant `−10` penalty. Now reset and hit **▶ Auto-drive** with **`V*(s)` turned ON**: watch the cyan values guide our empty taxi around the walls to Yellow `(4,0)`—and **watch the exact frame Pickup hits**: the entire city grid flips to emerald green and pulls the taxi straight to Green `(0,4)` for `+20`!
>
> Now over to Chapter 3—**Nishan**, take it away!"

---

## CHAPTER 03 — NISHAN (Roll No. 5) | Slides 15–20 (`10:45 – 16:00`)
*(Cinematic Chapter 03 interstitial triggers automatically on advancing to Slide 15!)*

### Slide 15 · The Animated Learning Race (Real 2,500-Ep Curves!) (`10:45 – 11:55`)
**[SHOWSTOPPER LIVE CUE: Hit `▶ Play race` on `Return`, pause to watch cyan DQN cross `+5.0` at ep `596`, then switch tab to `Penalties` and drag/replay to show emerald SAC drop straight to zero!]**
> "Thank you, Ashim! I'm **Nishan**, Roll Number **5**, and instead of showing you a static chart, let's **replay our actual 2,500 training episodes live**!
>
> Hit **Play race** on **Return**: watch cyan **DQN** break out of the pack early, crossing `+5.0` at **episode 596**, followed by emerald **SAC** at **808**, gold **SARSA** at **944**, and violet **PPO** at **1,023**.
>
> Now switch to **Penalties**: look at emerald **SAC** starting far below everyone else and crashing straight to zero, finishing with just **1,486** total illegal moves versus **10,908** for SARSA."

### Slide 16 · Final Benchmark Leaderboard (`11:55 – 12:45`)
> "Slide 16 summarizes all exact numbers from our notebook across 2,500 training and 500 unseen greedy test episodes.
>
> We have two unmistakable winners: **DQN** takes speed (`596` eps — `37%` faster than SARSA, `42%` faster than PPO) AND top greedy test return (**`+7.90 ± 2.64`**, `13.10` steps, `100.0%`). **Discrete SAC** takes safety (**`1,486` penalties** — `86%` fewer than SARSA, `79%` fewer than DQN) with `99.0%` test delivery, while **SARSA** hits `100.0%` (`+7.01`) and **PPO** hits `99.6%` (`+6.60`)."

### Slide 17 · When Learning Breaks: `α, γ, ε` Sensitivity (`12:45 – 13:40`)
> "Our 12×3 multi-seed sensitivity sweeps highlight three classic failure modes.
>
> In **Panel A**, `α = 0.20` is cleanest, `0.05` crawls, and aggressive **`α = 0.90` bounces around `−100`** as noisy targets overwrite prior knowledge.
>
> In **Panel B**, myopic **`γ = 0.50`** shrinks `+20` across 13 steps to `0.5¹³ ≈ 0.0001`—so Stage 1 never receives credit; `γ ∈ [0.95, 0.99]` fixes it.
>
> In **Panel C**, fixed `ε = 0.30` stays trapped below `−25` from perpetual `−10` errors, proving why **decaying `ε` (`1.0 → 0.01`)** is essential."

### Slide 18 · Inside the Agent's Head: `V*(s)` Heatmaps (`13:40 – 14:30`)
> "Inspecting the learned Q-table for mission `Y(4,0) → G(0,4)` confirms genuine spatial reasoning.
>
> In **Stage 1 (left)**, values climb smoothly to **`+5.2`** at Station Y with every arrow routing cleanly around walls into `PICKUP`. In **Stage 2 (center)**, the gradient flips completely toward **`+20.0`** at Station G—and dead-end `(4,1)` stays dark because optimal trajectories never enter it. On the right, starting from `(0,1)`, the agent drives the exact **14-step optimal path** with zero penalties."

### Slide 19 · Open-Source Repo, Limitations & Next Steps (`14:30 – 15:15`)
> "Every figure, animation, and experiment is 100% reproducible in three commands from our GitHub repo `RL_FINAL_MINIPROJECT` using our `rl_taxi` Conda environment.
>
> Being honest about limits: 500 discrete states fit cleanly in a table without generalization, and transitions are deterministic. Our clear next steps are scaling to 20×20+ grids with PyTorch Double-DQN / neural SAC, plus adding stochastic traffic delays and multi-taxi fleet dispatch."

### Slide 20 · Interactive Hiring Router & Closing (`15:15 – 16:00`)
**[LIVE CUE: Click `⚡ Speed` → `🛡 Safety` → `⚖ Stability` → `Show all` as you summarize each card!]**
> "So—which agent would you hire to drive your taxi?
>
> Care most about **Speed** under scarce experience? **Hire DQN** (`−37%` episodes, `+7.90` return). Care most about **Safety** where mistakes are costly? **Hire Discrete SAC** (`−86%` illegal penalties). Want **Stability** or clean simplicity? **Hire PPO or SARSA**.
>
> Scan the QR code on the bottom right for our complete GitHub code and notebook. On behalf of **Sijan Pandey (Roll No. 11)**, **Ashim Shrestha (Roll No. 22)**, and myself, **Nishan (Roll No. 5)**—thank you so much for watching!"

---

## Copy-Paste YouTube Description & Timestamped Chapters
```text
Autonomous Taxi Pickup & Drop-off: Comparative Analysis of SARSA, DQN, PPO, and Discrete SAC on Gymnasium Taxi-v4.

Team Members:
• Sijan Pandey (Roll No. 11)
• Ashim Shrestha (Roll No. 22)
• Nishan (Roll No. 5)

GitHub Repository: https://github.com/devsijanpandey12/RL_FINAL_MINIPROJECT

Chapters:
00:00 Mission Briefing & Team Intro
00:40 The Hook: 4-Agent Synchronized Rollout
01:30 Interactive Taxi-v4 City Grid
02:15 Agent–Environment RL Feedback Loop
02:45 Five Ingredients of the MDP (S, A, P, R, γ)
03:15 Formal MDP Details (500 States, 6 Actions, Rewards)
04:00 Controlled Pure-NumPy Ablation Design
04:45 Inside the Code (GitHub Notebook Highlights)
05:30 Agent 01: SARSA
06:20 Agent 02: DQN + Interactive Replay Demo
07:20 Agent 03: PPO + Interactive Trust Region
08:20 Agent 04: Discrete SAC + Live Temperature (τ) Lab
09:20 The RL Quadrant Map
09:55 Live Autonomous Taxi-v4 Simulator & V*(s) Flip
10:45 The Learning Race: 2,500 Real Training Episodes
11:55 Final Quantitative Leaderboard
12:45 When Learning Breaks: α, γ, ε Sensitivity
13:40 Inside the Agent's Head: V*(s) Heatmaps
14:30 Open-Source Repo, Limitations & Next Steps
15:15 Which Agent Would You Hire? & Closing
```
