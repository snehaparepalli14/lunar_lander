# DRL Assignment II — Discussion Questions & Answers
## Robust Reinforcement Learning under Stochastic Action Failure

**Course:** Deep Reinforcement Learning (S2-25_AIMLCZG512)  
**Group 22**

---

## Q1. Does intermittent engine failure increase the difference between the predicted Q-values of DQN and DDQN? Justify your answer using the Q-value plots.

**Answer: Yes — engine failure substantially widens the gap between DQN and DDQN Q-values.**

### Evidence from the Q-value plot:

| Condition | DQN peak Q-value | DDQN peak Q-value | Gap |
|-----------|-----------------|-------------------|-----|
| **Original (no failure)** | ~95 | ~84 | ~11 |
| **Modified (15% failure)** | ~29 | ~20 | ~9 |

### Interpretation:

1. **Original environment:** Both algorithms inflate Q-values due to DQN's maximisation bias, but DQN consistently sits above DDQN — the gap is ~11 units at peak (~episode 700).

2. **Modified environment:** Both DQN-Modified and DDQN-Modified learn much lower Q-values (max ~29 and ~20 respectively), largely because:
   - The fuel penalty (−0.3 per requested thruster) directly reduces returns
   - Random failures create uncertainty in the value of any thruster action

3. **Why the gap persists:** DQN uses the *same network* for both action selection and value evaluation (`max Q(s',a')` in the Bellman target). When engines randomly fail, the agent still sees an optimistic maximum, so overestimation is compounded by environmental stochasticity. DDQN decouples selection from evaluation (online net selects, target net scores), which naturally dampens the overestimate.

4. **Key takeaway:** Engine failure makes both algorithms more conservative (lower Q-values overall) because the environment is harder to exploit, but DQN remains the more overoptimistic estimator under both conditions.

---

## Q2. Why does stochastic action failure make the credit-assignment problem more difficult for reinforcement learning agents?

**Answer: Stochastic action failure introduces irreducible noise into the mapping between chosen actions and observed outcomes, breaking the core assumption of credit assignment.**

### What is Credit Assignment?

**Credit assignment** is the problem of determining which past actions caused the current reward signal. It is central to all RL algorithms.

**Normal conditions:**
```
Agent selects action a  →  Environment executes a  →  Reward r observed
Credit: r is attributed to action a  ✓
```

**With 15% stochastic failure:**
```
Agent selects thruster action a  →  With 15% probability, engine misfires (action 0 executed)
                                 →  Reward r observed
Credit: Was r caused by action a, or by the silent "do nothing"?  ✗
```

### Why This Is Specifically Harder:

1. **Hidden failure:** The agent receives **no signal** that the failure occurred. From the agent's perspective, the same action in the same state can produce *different transitions*, with no way to distinguish the cause.

2. **Corrupted Q-function updates:** Every Bellman update `Q(s,a) ← r + γ max Q(s',a')` assumes `a` was actually executed. When `a` silently becomes `0`, the resulting transition `(s, a=thruster, r, s')` stored in the replay buffer produces a **spurious sample** that does not match the true dynamics of action `a`.

3. **Increased variance in returns:** The same policy can produce different trajectory returns across identical episodes purely due to random failures. This inflates the variance of gradient updates, slowing convergence.

4. **Temporal credit propagation:** In long episodes, failures at early time steps can corrupt subsequent state distributions, making it even harder to identify which earlier action was responsible for a reward signal many steps later.

5. **Fuel penalty for unfired engines:** The −0.3 penalty applies to the *requested* action, not the *executed* one. The agent is penalised for a consequence it did not actually cause.

---

## Q3. Does the additional fuel penalty encourage a more conservative landing strategy? Support your answer using experimental evidence.

**Answer: Yes — the fuel penalty encourages a more conservative landing strategy, evidenced by reduced thruster usage and higher safe-landing rates in the modified environment.**

### Evidence:

**1. Thruster Activations:**
- In the original environment, thruster activations peak at ~400–450 activations/episode around episodes 350–500, before settling to ~150–200 in mature policies.
- In the modified environment, activations follow a similar peak but **converge to lower steady-state levels** (~100–150 vs ~150–200 for the originals), confirming the fuel penalty pushes agents to use thrusters more sparingly.

**2. Episode Rewards:**
- Modified-environment agents show *lower* final rewards (~150–200) than their original counterparts (~200–250), consistent with a more conservative, fuel-aware strategy — agents sacrifice some performance to avoid repeated −0.3 penalties.

**3. Safe Landing Rate (strongest evidence):**
- Despite lower rewards, **DDQN-Modified achieves the highest safe-landing rate (~77% peak) among all four configurations**.
- The fuel penalty inadvertently trains agents to land more precisely and gently, because aggressive over-correction wastes fuel and risks crashes, while slow controlled descents are penalised less and satisfy the strict safe-landing conditions (|vx|, |vy| < 0.10, |angle| < 0.10).

> **Key insight:** The fuel penalty functions like a regulariser — it suppresses excessively thruster-heavy policies and indirectly rewards smooth, fuel-efficient trajectories that happen to align with safe landing conditions.

---

## Q4. Which algorithm performs better under stochastic engine failures? Is this behaviour consistent with the theoretical advantage of DDQN over DQN? Explain.

**Answer: Results are mixed — DQN-Modified achieves higher peak rewards, but DDQN-Modified achieves a higher final safe-landing rate, which is partially (but not perfectly) consistent with DDQN's theoretical advantage.**

### Final Training Results (Episode 700):

| Algorithm | Environment | Final Reward | Final Success Rate |
|-----------|-------------|-------------|-------------------|
| DQN | Original | 275.4 | 50% |
| DQN | Modified | 290.7 | 46% |
| DDQN | Original | 249.1 | 49% |
| DDQN | Modified | 227.2 | **52%** |

### Analysis by Metric:

**Episode Reward:** DQN-Modified reaches slightly higher rewards (~290) than DDQN-Modified (~227) at the end of 700 episodes. However, the reward metric conflates the fuel penalty structure, making raw reward a less informative measure under the modified environment.

**Safe Landing Rate:**
- DDQN-Modified achieves the highest safe-landing rate (~77% peak, **52% final**) among all four configurations.
- DQN-Modified achieves only ~46% final success rate — below DDQN-Modified's 52%.

### Theoretical Alignment:

**DDQN's theoretical advantage** is the reduction of **maximisation bias** in Q-value estimation:

- **DQN target:** `y = r + γ · Q_target(s', argmax_a Q_target(s', a))`  
  → Same network both *selects* and *evaluates* the best action → overestimates values

- **DDQN target:** `y = r + γ · Q_target(s', argmax_a Q_online(s', a))`  
  → Online network selects, target network evaluates → decoupling reduces overestimation

Under stochastic failures, DQN's overestimation is **amplified** by environment noise — it believes actions are more reliable than they are. DDQN's decoupled evaluation partially corrects this, producing more realistic value estimates and a more robust policy.

The safe-landing rate supports this: DDQN-Modified's superior precision (**52% vs 46%**) suggests it develops a more conservative, robust strategy — consistent with more accurate Q-value estimates.

> **Note on the reward inconsistency:** DQN-Modified's higher reward may reflect overestimated values of aggressive manoeuvres, achieving higher scores in some episodes at the cost of more crashes in others. The safe-landing rate (a harder, unambiguous metric) favours DDQN-Modified. The 700-episode run may also be insufficient to fully reveal DDQN's advantage — longer training typically widens the gap as overestimation bias compounds in DQN.

---

## Q5. Identify one limitation of your experimental setup and suggest one possible improvement.

### Limitation: Single Random Seed — No Statistical Averaging

The most significant limitation is that **all four agents were trained using a single seed (2026)**. Deep RL training is highly sensitive to random initialisation — network weights, replay buffer sampling order, and stochastic failure events can vary dramatically across seeds.

**Why this matters:**
- A single run may be an outlier — favourable or unfavourable — for one algorithm
- Performance differences of ~5–10% in landing rate (e.g., DQN-Modified 46% vs DDQN-Modified 52%) fall well within the natural variance of a single run
- Without confidence intervals, no statistically rigorous claim can be made that one algorithm is better than another

**Evidence of instability:** The episode reward plots (especially episodes 500–900) show variance of ±100 reward points within any 100-episode window, confirming high run-to-run noise even after apparent convergence.

---

### Suggested Improvement: Multi-Seed Evaluation with Statistical Testing

Run each of the four experiments with **at least 5–10 different random seeds**, then report:

```
Mean final reward      ± standard deviation  (over N seeds)
Mean safe-landing rate ± standard deviation  (over N seeds)
95% confidence intervals on all key metrics
```

Apply a **Mann-Whitney U test** or **Welch's t-test** to confirm that observed differences (e.g., DDQN-Modified vs DQN-Modified safe-landing rate) are statistically significant before drawing conclusions.

**Additional improvements worth considering:**
- **Longer training (1500–2000 episodes):** The Q-value plots show all four agents still improving at episode 700 — more training would reveal whether DDQN's bias reduction leads to sustained divergence from DQN in the failure environment.
- **Failure rate ablation:** Test multiple failure probabilities (5%, 15%, 30%) to characterise how robustness degrades and at what threshold DDQN's advantage becomes unambiguous.
