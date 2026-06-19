from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SampleDecision:
    sample_idx: int
    rns_score: float
    rns_mean: float
    confidence: float
    current_label: any
    clean_ground_truth_label: any = None
    recommended_labels: List[any] = field(default_factory=list)
    decision: str = "UNKNOWN"
    corrected_label: None | Optional[any] = None
    rule_applied: int = 0