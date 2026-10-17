"""Fixed fitting-only search extracted from the verified notebook."""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from .metrics import choose_threshold
from .io import require, sha256_file, write_json

def select_model(split, base_config, run_dir, seed=42, target_fpr=0.05, enabled=True, evidence=None):
    fit, validation = split.fit, split.validation
    MODEL_CONFIG = dict(base_config)
    SEED, TARGET_FPR, CV_FOLDS = seed, target_fpr, 3
    SEARCH_C_VALUES = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    RUN_MODEL_SELECTION = enabled
    configuration = {}
    evidence = evidence or (lambda *args: None)
    selection_metadata = {'enabled': RUN_MODEL_SELECTION, 'scope': 'fitting partition only'}
    if RUN_MODEL_SELECTION:
        search_started = time.perf_counter()
        candidates = [{'C': float(value), 'class_weight': weight} for value in SEARCH_C_VALUES for weight in (None, 'balanced')]
        fold_results = [[] for _ in candidates]
        fitting_strata = fit['human_annotation'].astype(str) + ':' + fit['unsafe'].astype(str)
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        fold_manifest = []
        for fold_number, (inner_fit_idx, inner_check_idx) in enumerate(splitter.split(np.zeros(len(fit)), fitting_strata), start=1):
            inner_fit, inner_check = (fit.iloc[inner_fit_idx], fit.iloc[inner_check_idx])
            require(set(inner_fit['prompt_id']).isdisjoint(inner_check['prompt_id']), 'CV IDs overlap.')
            require(set(inner_fit['text_sha256']).isdisjoint(inner_check['text_sha256']), 'CV text overlaps.')
            require(set(inner_fit['prompt_id']).isdisjoint(validation['prompt_id']), 'Outer validation leaked into CV.')
            fold_manifest.append({'fold': fold_number, 'fit_ids': inner_fit['prompt_id'].tolist(), 'check_ids': inner_check['prompt_id'].tolist()})
            vectorizer = TfidfVectorizer(analyzer=MODEL_CONFIG['analyzer'], ngram_range=MODEL_CONFIG['ngram_range'], lowercase=MODEL_CONFIG['lowercase'], min_df=MODEL_CONFIG['min_df'], max_features=MODEL_CONFIG['max_features'], sublinear_tf=MODEL_CONFIG['sublinear_tf'], norm=MODEL_CONFIG['norm'])
            inner_x = vectorizer.fit_transform(inner_fit['user_input'])
            check_x = vectorizer.transform(inner_check['user_input'])
            check_human = inner_check['human_annotation'].to_numpy(dtype=bool)
            check_y = inner_check['unsafe'].to_numpy(dtype=int)
            require(np.unique(check_y[check_human]).size == 2, 'Human CV fold needs both classes.')
            for index, candidate in enumerate(candidates):
                classifier = LogisticRegression(solver='liblinear', l1_ratio=0.0, C=candidate['C'], class_weight=candidate['class_weight'], max_iter=MODEL_CONFIG['max_iter'], random_state=SEED)
                started = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    classifier.fit(inner_x, inner_fit['unsafe'])
                unsafe_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
                scores = classifier.predict_proba(check_x)[:, unsafe_index]
                benign = scores[check_human & (check_y == 0)]
                cutoff = choose_threshold(benign, TARGET_FPR)
                unsafe = check_human & (check_y == 1)
                tp = int(np.count_nonzero(scores[unsafe] >= cutoff))
                fp = int(np.count_nonzero(benign >= cutoff))
                fold_results[index].append({'fold': fold_number, 'tp': tp, 'unsafe': int(unsafe.sum()), 'fp': fp, 'benign': len(benign), 'threshold': cutoff, 'average_precision': float(average_precision_score(check_y[check_human], scores[check_human])), 'fit_seconds': time.perf_counter() - started})
            del inner_x, check_x, classifier, vectorizer
        leaderboard = []
        for candidate, results in zip(candidates, fold_results):
            leaderboard.append({**candidate, 'pooled_human_recall': sum((r['tp'] for r in results)) / sum((r['unsafe'] for r in results)), 'pooled_human_fpr': sum((r['fp'] for r in results)) / sum((r['benign'] for r in results)), 'mean_average_precision': float(np.mean([r['average_precision'] for r in results])), 'folds': results})
        chosen = max(leaderboard, key=lambda row: (row['pooled_human_recall'], row['mean_average_precision'], -row['C'], row['class_weight'] is None))
        MODEL_CONFIG.update({key: chosen[key] for key in ('C', 'class_weight')})
        selection_metadata = {'enabled': True, 'scope': 'fitting partition only', 'folds': CV_FOLDS, 'candidate_count': len(candidates), 'classifier_fits': len(candidates) * CV_FOLDS, 'ranking': 'pooled human recall at <=5% fold FPR, then mean AP, smaller C, no weighting', 'selected_parameters': {key: chosen[key] for key in ('C', 'class_weight')}, 'selected_cv_recall': chosen['pooled_human_recall'], 'selected_cv_average_precision': chosen['mean_average_precision'], 'search_seconds': time.perf_counter() - search_started}
        write_json(run_dir / 'model-selection.json', {'selection': selection_metadata, 'leaderboard': leaderboard, 'fold_membership': fold_manifest})
        selection_metadata['search_artifact_sha256'] = sha256_file(run_dir / 'model-selection.json')
        table = pd.DataFrame([{key: row[key] for key in ('C', 'class_weight', 'pooled_human_recall', 'pooled_human_fpr', 'mean_average_precision')} for row in leaderboard]).sort_values(['pooled_human_recall', 'mean_average_precision'], ascending=False)
    configuration['model_selection'] = selection_metadata
    evidence('model_selection_completed', selection_metadata)
    return MODEL_CONFIG, selection_metadata
