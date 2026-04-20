# Later (explicitly deferred until after the 2-day demo)

Do NOT start any of these during the sprint. They are recorded here so the architect chat doesn't relitigate them.

1. **Post-quant Mimi dataset (`audio_rt_aligned_q/`)** + short bridge fine-tune + live path consuming `main_latents` directly (skips PCM roundtrip). Potential quality win and ~1 Mimi decoder worth of GPU time saved per chunk. At least 1 day of compute.
2. **Mouth-subspace learning from training motion latents** (PCA / linear regression against mouth-open estimate). Extends task 002's quick fix into a more principled version.
3. **iMF branch 1-step generation** — entire separate repo, blocked by FM branch stabilisation.
4. **GitHub repo + proper CI**. Plan after the demo. For now we use local git + rsync.
5. **Renderer fine-tune for higher resolution / better teeth.** Out of scope.
6. **TensorBoard / W&B integration on all training runs.** Add after demo.
7. **Custom pod MCP server** (high-level verbs: `pod.run_training`, `pod.training_status`). Nice to have, not worth the setup cost pre-demo.
