from dataclasses import dataclass

@dataclass
class SampleNoiseProfile:
    sample_idx: int
    noise_score: float
    noise_rank: int
    noise_category: str
    simulated_action: str
    confidence: float
    was_ever_in_NF: bool
    final_decision: str
    source_label:any
    working_label: any
    true_label: any = None
    is_truly_noisy: bool = None
