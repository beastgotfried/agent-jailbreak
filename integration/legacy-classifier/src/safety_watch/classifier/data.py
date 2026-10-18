"""Verified ToxicChat loading and exact reproduction of the original partition policy."""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .io import require, sha256_file, write_json

REQUIRED_COLUMNS = ["conv_id", "user_input", "human_annotation", "toxicity", "jailbreaking"]

@dataclass(repr=False)
class LoadedData:
    train: pd.DataFrame
    train_path: Path
    test_path: Path
    metadata: dict

@dataclass(repr=False)
class DataSplit:
    fit: pd.DataFrame
    validation: pd.DataFrame
    excluded: pd.DataFrame
    partitions: dict
    summary: dict
    digest: str

def load_data(data_root, evidence=None):
    DATA_ROOT = Path(data_root)
    evidence = evidence or (lambda *args: None)
    manifest = json.loads((DATA_ROOT / 'download-manifest.json').read_text())
    entries = [entry for entry in manifest if entry.get('repo') == 'lmsys/toxic-chat']
    require(len(entries) == 1, 'Expected one ToxicChat manifest entry.')
    dataset_entry = entries[0]
    raw_root = DATA_ROOT / 'raw' / 'toxicchat0124'

    def manifest_file(filename):
        matches = [item for item in dataset_entry['files'] if item['path'] == 'data/0124/' + filename]
        require(len(matches) == 1, 'Expected unique ToxicChat 0124 file.')
        item = matches[0]
        path = raw_root / item['path']
        require(path.is_file(), 'ToxicChat file missing.')
        digest = sha256_file(path)
        require(digest == item['sha256'], 'Dataset SHA-256 mismatch; do not train.')
        require(path.stat().st_size == item['bytes'], 'Dataset byte count mismatch.')
        return (path, digest)
    train_path, train_digest = manifest_file('toxic-chat_annotation_train.csv')
    test_path, test_digest = manifest_file('toxic-chat_annotation_test.csv')
    required_columns = ['conv_id', 'user_input', 'human_annotation', 'toxicity', 'jailbreaking']
    train = pd.read_csv(train_path, usecols=required_columns, dtype={'conv_id': 'string', 'user_input': 'string'})
    require(not train[required_columns].isna().any().any(), 'Missing required training fields.')
    require(train['user_input'].str.strip().str.len().gt(0).all(), 'Empty training prompts.')
    for label in ('toxicity', 'jailbreaking'):
        require(train[label].isin([0, 1]).all(), 'Labels must be binary 0/1.')
    train['human_annotation'] = train['human_annotation'].astype(str).map({'True': True, 'False': False})
    require(train['human_annotation'].notna().all(), 'Invalid annotation flag.')
    train['unsafe'] = (train['toxicity'].eq(1) | train['jailbreaking'].eq(1)).astype(int)
    train['prompt_id'] = ['toxicchat0124:train:' + str(index) for index in range(len(train))]
    train['text_sha256'] = train['user_input'].map(lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest())
    dataset_metadata = {'repo': 'lmsys/toxic-chat', 'revision': dataset_entry['revision'], 'train_sha256': train_digest, 'test_sha256': test_digest, 'official_training_rows': len(train), 'license': 'CC-BY-NC-4.0'}
    evidence('data_verified', dataset_metadata)
    return LoadedData(train, train_path, test_path, dataset_metadata)

def prepare_split(data, run_dir, seed=42, validation_fraction=0.20, evidence=None):
    evidence = evidence or (lambda *args: None)
    train, test_path = data.train, data.test_path
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    SEED, VALIDATION_FRACTION = seed, validation_fraction
    test_text = pd.read_csv(test_path, usecols=['user_input'], dtype={'user_input': 'string'})
    require(test_text['user_input'].notna().all(), 'Missing test text.')
    test_hashes = set(test_text['user_input'].map(lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest()))
    require(train.groupby('text_sha256')['unsafe'].nunique().le(1).all(), 'Conflicting duplicate training targets.')
    overlap = train['text_sha256'].isin(test_hashes)
    audit_columns = ['prompt_id', 'conv_id', 'text_sha256', 'human_annotation', 'unsafe']
    excluded_overlap = train.loc[overlap, audit_columns].copy()
    excluded_overlap['reason'] = 'exact_test_text_overlap'
    eligible = train.loc[~overlap].sort_values(['human_annotation', 'conv_id', 'prompt_id'], ascending=[False, True, True], kind='stable').copy()
    duplicate = eligible.duplicated('text_sha256', keep='first')
    excluded_duplicate = eligible.loc[duplicate, audit_columns].copy()
    excluded_duplicate['reason'] = 'duplicate_training_text'
    excluded = pd.concat([excluded_overlap, excluded_duplicate], ignore_index=True)
    eligible = eligible.loc[~duplicate].sort_values('prompt_id', kind='stable').reset_index(drop=True)
    strata = eligible['human_annotation'].astype(str) + ':' + eligible['unsafe'].astype(str)
    require(strata.value_counts().ge(2).all(), 'Split stratum has fewer than two records.')
    fit_idx, validation_idx = train_test_split(np.arange(len(eligible)), test_size=VALIDATION_FRACTION, random_state=SEED, stratify=strata)
    fit, validation = (eligible.iloc[fit_idx].copy(), eligible.iloc[validation_idx].copy())
    require(set(fit['prompt_id']).isdisjoint(validation['prompt_id']), 'Fit/validation IDs overlap.')
    require(set(fit['text_sha256']).isdisjoint(validation['text_sha256']), 'Fit/validation text overlaps.')
    require(set(eligible['text_sha256']).isdisjoint(test_hashes), 'Eligible training overlaps test.')
    require(fit['unsafe'].nunique() == 2 and validation['unsafe'].nunique() == 2, 'Each partition needs both classes.')
    partitions = {'fit': fit[audit_columns].to_dict(orient='records'), 'validation': validation[audit_columns].to_dict(orient='records'), 'excluded': excluded.to_dict(orient='records')}
    write_json(run_dir / 'partitions.json', partitions)
    partition_digest = sha256_file(run_dir / 'partitions.json')
    split_summary = {'cross_boundary_excluded': int(overlap.sum()), 'within_training_excluded': int(duplicate.sum()), 'eligible': len(eligible), 'fit': len(fit), 'validation': len(validation), 'partitions_sha256': partition_digest}
    evidence('partitions_locked', split_summary)
    del test_text, test_hashes
    return DataSplit(fit, validation, excluded, partitions, split_summary, partition_digest)
