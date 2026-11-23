"""ReGAP: the Reliance Gap metric for temporally grounded video reasoning.

Given, for every evaluation item, the model's probability for its own answer
  * with the full video                      -> p_full
  * after deleting its predicted span        -> p_span_deleted
  * after deleting K random blocks of the same length (averaged) -> p_random_deleted
ReGAP = E[ logit(p_random_deleted) - logit(p_span_deleted) ] / E[ |logit(p_full)| ].
"""
from .metric import (CLIP, HIT_IOU, logit, mean_random, item_status, item_gap, span_iou, regap, accuracy,
                     summarize, bootstrap)

__all__ = ["CLIP", "HIT_IOU", "logit", "mean_random", "item_status", "item_gap", "span_iou", "regap", "accuracy",
           "summarize", "bootstrap"]
