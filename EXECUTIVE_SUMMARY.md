## 🧾 Executive Summary: Sci‑Fi Movie Recommendation for Cold-Start Users

**Question.** Does the choice of collaborative-filtering algorithm materially change recommendation quality for Sci‑Fi movies, and does any of that quality survive for cold-start users (≤ 6 ratings)?

**Approach.** Three recommenders (`KNN`, `SVD`, `NMF`) were compared against a bias-only baseline ($\mu + b_u + b_i$) and a popularity baseline on a random 20,000-user sample of MovieLens 32M (582,670 Sci‑Fi ratings; 14,009 active and 5,991 cold-start users), with bootstrap confidence intervals for the cold-start analysis.

**Key findings**

- **Rating accuracy: `SVD` is best, but the gain is modest.** `SVD` reaches a test RMSE of 0.813 vs. 0.843 for `NMF`, and lowers RMSE by about 6% relative to the bias-only baseline on the validation set (0.820 vs. 0.871).
- **Ranking: nothing beats popularity.** Top-10 precision is 0.035 (`SVD`) and 0.028 (`NMF`) vs. 0.136 for a most-popular baseline; recall@10 is 0.075 and 0.067 vs. 0.288.
- **`KNN` did not outperform the simple baselines.** Once ratings were centered, `KNN` fell below a per-user-average predictor (R² = −0.14). This is tentative, because `KNN` was evaluated under a different protocol and a Cosine variant scored lower error.
- **Cold-start: the profile matters more than the model.** Even one or two ratings beat using none. `SVD` overtakes the baseline in point estimate at roughly three ratings, but the difference is statistically distinguishable only at five (borderline). `NMF` never beats the baseline in this configuration.

**Recommendation.** Use a popularity or bias-only recommender for new users, and treat `SVD` as a candidate once a user has about three to five ratings, pending validation on more data. Choose rankers on ranking metrics (precision@k, recall@k), not RMSE alone.

**Caveats.** Results are for a Sci‑Fi-only, randomly selected 20,000-user sample; cold-start users have 1–6 ratings (truly unseen users are not modeled); per-size confidence intervals are wide (about ±0.09 MSE); the genre-granularity result is an exploratory single-user case study.
