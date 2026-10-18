"""Canonical learned-state identity for the selected scikit-learn pipelines.

CountVectorizer caches id(stop_words) in _stop_words_id. That process address
is not learned state and can change pickle bytes between otherwise exact fits.
Only that field, on a TfidfVectorizer, is excluded. Distributed file hashes stay
mandatory; this comparison does not authorize substitute deployment artifacts.
"""
import hashlib
import json

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def canonical(value):
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, np.generic):
        return ["numpy_scalar", str(value.dtype), canonical(value.item())]
    if isinstance(value, np.dtype):
        return ["dtype", value.str]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("object_array_is_not_supported_model_state")
        return ["array", value.dtype.str, list(value.shape),
                hashlib.sha256(value.tobytes(order="C")).hexdigest()]
    if sparse.issparse(value):
        matrix = value.tocsr(copy=True)
        matrix.sort_indices()
        return ["sparse_csr", list(matrix.shape), canonical(matrix.data),
                canonical(matrix.indices), canonical(matrix.indptr)]
    if isinstance(value, type):
        return ["type", value.__module__, value.__qualname__]
    if isinstance(value, dict):
        items = [[canonical(k), canonical(v)] for k, v in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return ["dict", items]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical(v) for v in value]]
    if isinstance(value, (set, frozenset)):
        items = [canonical(v) for v in value]
        items.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return [type(value).__name__, items]
    if type(value).__module__.startswith("sklearn.") and hasattr(value, "__dict__"):
        attributes = dict(value.__dict__)
        if isinstance(value, TfidfVectorizer):
            attributes.pop("_stop_words_id", None)
        return ["estimator", type(value).__module__, type(value).__qualname__, canonical(attributes)]
    raise ValueError("unsupported_model_state_type")


def learned_state_sha256(model):
    payload = json.dumps(canonical(model), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()
