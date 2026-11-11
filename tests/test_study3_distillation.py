from __future__ import annotations

import numpy as np
import torch

from ts.future_work.film_double_buffer import (
    DoubleBufferedContextStore,
    FiLMConditionedStudent1DCNN,
)
from ts.study3.distillation import (
    ReasoningDistillationLoss,
    StudentDistillationTrainer,
    verify_property_5_1_orthogonality,
)
from ts.study3.student_model import Student1DCNN
from ts.study3.teacher_signal import (
    FinBERTReasoningEncoder,
    build_aligned_distillation_dataset,
    compute_teacher_soft_logits_and_probs,
)


def test_student_1dcnn_appendix_c_and_property_5_1() -> None:
    model = Student1DCNN(input_dim=40, hidden_dim=128, action_dim=5, llm_dim=768)
    x = torch.randn(16, 40, 100)
    logits, val, h_proj = model(x)
    assert logits.shape == (16, 5)
    assert val.shape == (16, 1)
    assert h_proj.shape == (16, 768)
    # Projection head has NO ReLU at output -> contains both positive and negative coordinates
    assert (h_proj < 0).any() and (h_proj > 0).any()

    # Online path returns greedy actions & probabilities without running critic or projection_head
    acts, probs = model.act_online(x)
    assert acts.shape == (16,)
    assert probs.shape == (16, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)

    # Property 5.1 (Eq. V.10): Orthogonal gradient H_hat · ∂L_reasoning/∂H' == 0 & norm bound <= 1/||H'||
    e_reason = torch.randn(16, 768)
    prop = verify_property_5_1_orthogonality(h_proj, e_reason)
    assert prop["max_abs_dot_product"] < 1e-5
    assert prop["max_grad_norm_minus_bound"] <= 1e-5


def test_distillation_autograd_intact_and_lambda3_sweep() -> None:
    rng = np.random.default_rng(11)
    n_bars = 8
    per_bar_wins = [
        [rng.standard_normal((40, 100)).astype(np.float32) for _ in range(10)]
        for _ in range(n_bars)
    ]
    w_finals = np.linspace(-1.0, 1.0, num=n_bars, dtype=np.float32)
    texts = [f"[Meta PM Consensus] Bar {i} rationale." for i in range(n_bars)]
    rets = rng.normal(0, 0.004, size=n_bars).astype(np.float32)
    costs = np.full(n_bars, 0.001, dtype=np.float32)

    z_t, p_t = compute_teacher_soft_logits_and_probs(0.5, sigma_w=0.25, tau=2.0)
    assert z_t.shape == (5,)
    assert int(np.argmax(p_t)) == 3  # +0.5 corresponds to index 3!

    finbert = FinBERTReasoningEncoder(offline_hash_mode=True)
    pack = build_aligned_distillation_dataset(
        per_bar_lob_windows=per_bar_wins,
        w_final_per_bar=w_finals,
        consensus_texts_per_bar=texts,
        next_bar_returns=rets,
        one_way_cost_rates=costs,
        finbert_encoder=finbert,
    )
    assert pack.lob_windows.shape == (80, 40, 100)  # 8 bars * 10 windows/bar!

    # Verify Conv1d weights actually update (fixing the `.item()` zero-grad bug in old repo!)
    trainer = StudentDistillationTrainer(
        lambda_reasoning=0.2, epochs=2, batch_size=32, device=torch.device("cpu")
    )
    init_m = Student1DCNN()
    torch.manual_seed(42)
    init_conv1 = init_m.conv1.weight.detach().clone()

    trained_m, hist = trainer.fit(pack)
    assert len(hist) == 2
    assert not torch.allclose(trained_m.conv1.weight.detach().cpu(), init_conv1)

    # Verify loss weight sum constraint
    loss_fn = ReasoningDistillationLoss(lambda_task=0.25, lambda_policy=0.25, lambda_reason=0.50)
    assert abs(loss_fn.lambda_task + loss_fn.lambda_policy + loss_fn.lambda_reason - 1.0) < 1e-6


def test_appendix_d_double_buffer_and_stale_fallback() -> None:
    store = DoubleBufferedContextStore(hidden_dim=128, max_staleness_sec=3600.0)
    gamma_w = np.full(128, 1.5, dtype=np.float32)
    beta_w = np.full(128, 0.2, dtype=np.float32)
    slot = store.teacher_write(gamma_w, beta_w, now_ts=10_000.0)
    assert slot == 1

    g_ok, b_ok, is_stale = store.student_read(now_ts=11_000.0)
    assert not is_stale
    assert np.allclose(g_ok, 1.5) and np.allclose(b_ok, 0.2)

    # Exceed 1-hour staleness -> must trigger defensive fallback (gamma=0, beta=-10)
    g_st, b_st, is_stale2 = store.student_read(now_ts=15_000.0)
    assert is_stale2
    assert np.allclose(g_st, 0.0) and np.allclose(b_st, -10.0)

    film_student = FiLMConditionedStudent1DCNN()
    acts, _, stale_flag = film_student.act_with_context(torch.randn(2, 40, 100), store, now_ts=15_000.0)
    assert stale_flag and acts.shape == (2,)
    store.close()
