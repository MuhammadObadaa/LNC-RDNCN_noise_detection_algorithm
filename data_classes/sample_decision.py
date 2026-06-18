from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SampleDecision:
    sample_idx: int
    rns_score: float
    rns_mean: float
    confidence: float
    current_label: int
    clean_ground_truth_label: int = None
    recommended_labels: List[int] = field(default_factory=list)
    decision: str = "UNKNOWN"
    corrected_label: None | Optional[int] = None
    rule_applied: int = 0