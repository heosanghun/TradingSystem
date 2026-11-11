"""Study 3 (Ch. V & Appendix C): Dual-Process Reasoning Knowledge Distillation."""

from ts.study3.distillation import (
    ReasoningDistillationLoss,
    StudentDistillationTrainer,
    verify_property_5_1_orthogonality,
)
from ts.study3.student_model import Student1DCNN
from ts.study3.teacher_signal import (
    FinBERTReasoningEncoder,
    TeacherSignalPack,
    build_aligned_distillation_dataset,
    compute_teacher_soft_logits_and_probs,
)

__all__ = [
    "ReasoningDistillationLoss",
    "StudentDistillationTrainer",
    "verify_property_5_1_orthogonality",
    "Student1DCNN",
    "FinBERTReasoningEncoder",
    "TeacherSignalPack",
    "build_aligned_distillation_dataset",
    "compute_teacher_soft_logits_and_probs",
]
