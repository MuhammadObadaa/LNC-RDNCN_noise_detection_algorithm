from dataclasses import dataclass, field
from typing import List
from data_classes import sample_decision

@dataclass
class IterationRecord:
    iteration: int
    n_active: int
    NF_size: int
    rns_mean: float
    n_retained: int
    n_corrected: int
    n_deleted: int
    decisions: List[sample_decision.SampleDecision] = field(default_factory=list)
    