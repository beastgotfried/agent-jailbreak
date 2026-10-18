"""Threshold selection and metrics retain their original denominators and tie policy."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from .io import require
def choose_threshold(benign, maximum_fpr):
    benign = np.asarray(benign, dtype=float)
    require(benign.size > 0 and np.isfinite(benign).all(), 'Threshold needs finite benign scores.')
    candidates = np.unique(np.concatenate(([0.0], benign, np.nextafter(benign, np.inf), [np.nextafter(1.0, np.inf)])))
    allowed = int(np.floor(maximum_fpr * benign.size + 1e-12))
    for candidate in candidates:
        if int(np.count_nonzero(benign >= candidate)) <= allowed:
            return float(candidate)
    raise RuntimeError('No threshold satisfies the FPR target.')

def metrics(y, scores, decision_threshold):
    y, scores = (np.asarray(y, dtype=int), np.asarray(scores, dtype=float))
    require(len(y) == len(scores) and len(y) > 0, 'Invalid metric sizes.')
    require(np.unique(y).size == 2, 'Metrics require both classes.')
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), 'Invalid metric scores.')
    predicted = (scores >= decision_threshold).astype(int)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return {'records': len(y), 'benign': tn + fp, 'unsafe': tp + fn, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision': float(precision_score(y, predicted, zero_division=0)), 'recall': float(recall_score(y, predicted, zero_division=0)), 'f1': float(f1_score(y, predicted, zero_division=0)), 'fpr': fp / (tn + fp), 'false_negative_rate': fn / (tp + fn), 'average_precision': float(average_precision_score(y, scores))}

def validation_report(frame, scores, threshold):
    human = frame["human_annotation"].to_numpy(dtype=bool)
    y = frame["unsafe"].to_numpy(dtype=int)
    return {
        "human_annotated": metrics(y[human], scores[human], threshold),
        "full_validation_supplementary": metrics(y, scores, threshold),
    }
