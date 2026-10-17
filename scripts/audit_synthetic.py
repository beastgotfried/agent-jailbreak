"""Independent arithmetic over stored evidence. No model fit or application imports."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import sys

root = Path(sys.argv[1])
assert platform.machine() == 's390x' or os.environ.get('SAFETY_WATCH_ALLOW_LOCAL') == '1'
bench = root/'benchmark'
report = json.loads((bench/'report.json').read_text())
cards = json.loads((root/'models/model-cards.json').read_text())
plan = json.loads((bench/'evaluation-plan.json').read_text())
healthy = json.loads((bench/'healthy-scores.json').read_text())
threshold = cards['guardrail_threshold']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tid(text):
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def decision(g):
    return 'refuse' if g >= threshold else 'allow'


def transform(mode, g, c):
    if mode in ('scale', 'common_mode_scale'):
        g *= .25
    elif mode == 'affine':
        g = .55*g + .12
    elif mode == 'square':
        g *= g
    elif mode == 'clipped':
        g = min(g, .15)
    elif mode in ('constant', 'persistent_restart_fault'):
        g = .1
    elif mode == 'inverted':
        g = 1-g
    if mode == 'common_mode_scale':
        c *= .25
    return g, c


def pair(pid, mode, chain=()):
    g, c = transform(mode, healthy[pid]['guardrail_score'], healthy[pid]['canary_score'])
    for slope, offset in chain:
        g = max(0., min(1., slope*g + offset))
    return g, c


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-10), 'numeric_mismatch'


rows = {}
family_splits, semantic_splits, cases, counts = defaultdict(set), defaultdict(set), defaultdict(set), Counter()
for split, expected in [('train', 70000), ('validation', 10000), ('test', 20000)]:
    n = 0
    with (root/'data'/f'{split}.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            assert r['prompt_id'] == tid(r['text']) and r['prompt_id'] not in rows
            assert r['split'] == split and r['label'] in (0, 1)
            rows[r['prompt_id']] = {k:v for k,v in r.items() if k != 'text'}
            family_splits[r['family']].add(split)
            semantic_splits[r['semantic_group']].add(split)
            cases[(r['family'], r['target_index'], r['context_index'], r['frame_index'])].add(r['label'])
            counts[(split, r['label'])] += 1
            n += 1
    assert n == expected
    assert counts[(split, 0)] == counts[(split, 1)] == expected//2
assert len(rows) == 100000 and len(family_splits) == 100
assert all(len(v) == 1 for v in family_splits.values())
assert all(len(v) == 1 for v in semantic_splits.values())
assert all(v == {0, 1} for v in cases.values())
assert len(cases) == 50000
assert len(plan['endpoint_ids']) == 1000
for field, role in [('fit_ids', 'probe_fit'), ('verification_ids', 'probe_verification')]:
    assert all(rows[pid]['role'] == role and rows[pid]['split'] == 'validation' for pid in plan[field])
assert set(plan['fit_ids']).isdisjoint(plan['verification_ids'])
assert (set(plan['fit_ids']) | set(plan['verification_ids'])).isdisjoint(cards['policy']['calibration_ids'])
assert all(rows[pid]['split'] == 'test' for pid in plan['endpoint_ids'])


def service(ids, decisions, stored):
    assert len(ids) == len(decisions) == stored['n']
    correct = 0
    for label in (0, 1):
        subset = [d for pid,d in zip(ids, decisions) if rows[pid]['label'] == label]
        expected = {'n':len(subset), 'allowed':subset.count('allow'), 'refused':subset.count('refuse'),
                    'review':subset.count('review'), 'held':subset.count('hold'), 'unavailable':subset.count('unavailable')}
        assert expected == stored['by_class'][str(label)]
        correct += expected['allowed' if label == 0 else 'refused']
    close(correct/len(ids), stored['operational_accuracy'])
    if 'correct_automatic_decisions' in stored:
        assert correct == stored['correct_automatic_decisions']
    if 'correct' in stored:
        assert correct == stored['correct']
    close(sum(d in ('allow','refuse') for d in decisions)/len(ids), stored['automatic_coverage'])


ordered = sorted(plan['fit_ids'], key=lambda pid:(healthy[pid]['guardrail_score'],pid))
panel = [ordered[i*(len(ordered)-1)//7] for i in range(8)]
all_ids, all_decisions = [], {'A':[], 'B':[]}
fault_ids, fault_decisions = [], {'A':[], 'B':[]}
control_ids, control_decisions = [], {'A':[], 'B':[]}
costs, restored, held, confirmed, false_confirmations = Counter(), 0, 0, 0, 0
for scenario in plan['scenarios']:
    scenario_ids, scenario_decisions = [], {'A':[], 'B':[]}
    group = report['aggregate'][scenario]
    for seed in plan['seeds']:
        prefix = f'{scenario}-{seed}'
        summary = json.loads((bench/(prefix+'-summary.json')).read_text())
        events = [json.loads(line) for line in (bench/(prefix+'-events.jsonl')).read_text().splitlines()]
        ids = [e['prompt_id'] for e in events]
        assert len(ids) == len(set(ids)) == 512 and identity(ids) == summary['input_digest']
        assert all(rows[pid]['split'] == 'test' for pid in ids)
        a_mode = b_mode = None
        chain, generation, has_hold = [], 0, False
        for index, event in enumerate(events):
            assert index == event['index'] and event['label'] == rows[event['prompt_id']]['label']
            if index in summary['onsets']:
                a_mode = b_mode = ('constant' if scenario == 'scale_then_constant' and index == 320 else
                                  'scale' if scenario in ('late_scale','scale_then_constant') else scenario)
            close(event['A_guardrail_score'], pair(event['prompt_id'], a_mode)[0])
            assert event['A_decision'] == decision(event['A_guardrail_score'])
            health = event['health']
            assert health['checked'] == ((index+1)%16 == 0)
            assert health['generation'] == generation
            if health['checked']:
                values = [pair(pid, b_mode, chain) for pid in panel]
                ge = max(abs(v[0]-healthy[pid]['guardrail_score']) for pid,v in zip(panel, values))
                ce = max(abs(v[1]-healthy[pid]['canary_score']) for pid,v in zip(panel, values))
                close(ge, health['guardrail_error']); close(ce, health['canary_error'])
                assert health['changed'] == (max(ge,ce)>1e-8)
            response = event['response']
            if response:
                assert response['response_index'] == index
                if response['applied']:
                    generation += 1
                    if response['method'] == 'trusted_scorer_restart':
                        b_mode, chain = None, []
                    else:
                        attempt = next(v for v in response['attempts'] if v['method']=='paired_affine_correction' and v.get('verification',{}).get('accepted'))
                        assert attempt['verification']['pair_verified']
                        chain.append((attempt['slope'], attempt['intercept']))
                elif response['status'] != 'no_probe_degradation':
                    has_hold = True
            assert generation == event['generation']
            if event['B_scores'] is not None:
                assert not has_hold
                g,c = pair(event['prompt_id'], b_mode, chain)
                close(g, event['B_scores']['guardrail_score']); close(c, event['B_scores']['canary_score'])
                assert event['B_decision'] == decision(event['B_scores']['guardrail_score'])
            else:
                assert has_hold and event['B_decision'] == 'hold'
            assert event['B_execute'] == (event['B_decision'] == 'allow')
        assert has_hold == summary['held'] and generation == summary['generation']
        for arm in ('A','B'):
            decisions = [e[arm+'_decision'] for e in events]
            service(ids, decisions, summary[arm+'_stream'])
            scenario_decisions[arm] += decisions
            all_decisions[arm] += decisions
            (fault_decisions if summary['onsets'] else control_decisions)[arm] += decisions
        scenario_ids += ids; all_ids += ids
        (fault_ids if summary['onsets'] else control_ids).extend(ids)
        endpoint = json.loads((bench/(prefix+'-endpoint.json')).read_text())
        final_ids = [r['prompt_id'] for r in endpoint]
        assert final_ids == plan['endpoint_ids']
        ag, bg, bc = [], [], []
        for e in endpoint:
            expected_a = pair(e['prompt_id'],a_mode)[0]
            expected_g, expected_c = pair(e['prompt_id'],b_mode,chain)
            close(expected_a,e['A_guardrail_score']); close(expected_g,e['B_scores']['guardrail_score'])
            close(expected_c,e['B_scores']['canary_score'])
            ag.append(expected_a); bg.append(expected_g); bc.append(expected_c)
        service(final_ids, [decision(g) for g in ag], summary['A_final'])
        service(final_ids, ['hold']*len(final_ids) if has_hold else [decision(g) for g in bg], summary['B_final'])
        service(final_ids, [decision(g) for g in bg], summary['B_backend_final'])
        ge = max(abs(g-healthy[pid]['guardrail_score']) for pid,g in zip(final_ids,bg))
        ce = max(abs(c-healthy[pid]['canary_score']) for pid,c in zip(final_ids,bc))
        mismatch = sum(decision(g)!=decision(healthy[pid]['guardrail_score']) for pid,g in zip(final_ids,bg))
        close(ge,summary['final_guardrail_error']); close(ce,summary['final_canary_error'])
        assert mismatch == summary['final_decision_mismatches']
        expected_restore = bool(summary['onsets'] and generation and ge<=1e-8 and ce<=1e-8 and mismatch==0)
        assert expected_restore == summary['restored_backend']
        assert (expected_restore and not has_hold) == summary['restored_service']
        for d in summary['causal_detection']:
            stop = next((n for n in summary['onsets'] if n>d['fault_index']),512)
            inc = next((i for i in summary['incidents'] if d['fault_index']<=i['confirmed_index']<stop),None)
            assert d['confirmed_index'] == (inc['confirmed_index'] if inc else None)
            if inc:
                assert d['confirmation_delay'] == inc['confirmed_index']-d['fault_index']
                assert d['response_index'] == inc.get('response_index')
                confirmed += 1
        restored += summary['restored_service']; held += has_hold
        false_confirmations += summary['pre_fault_confirmations']
        costs.update(summary['costs'])
    for arm in ('A','B'):
        service(scenario_ids, scenario_decisions[arm], group[arm])
for scope, ids, decisions in [('overall',all_ids,all_decisions),('fault_only',fault_ids,fault_decisions),('no_fault',control_ids,control_decisions)]:
    for arm in ('A','B'):
        service(ids, decisions[arm], report[scope][arm])
assert dict(costs) == report['costs']
assert costs['A:live:inputs'] == len(all_ids)
assert costs['B:health:inputs'] == len(all_ids)//16*8
assert report['runtime_model_inputs']['A_canary'] == 0
for name, metric in report['healthy_test'].items():
    confusion = Counter()
    for pid, r in rows.items():
        if r['split'] != 'test':
            continue
        positive = healthy[pid][name+'_score'] >= metric['threshold']
        confusion['tp' if positive and r['label'] else 'fp' if positive else 'fn' if r['label'] else 'tn'] += 1
    assert dict(confusion) == {k:v for k,v in metric['confusion'].items() if v}
    tp,tn,fp,fn = [confusion[k] for k in ('tp','tn','fp','fn')]
    close((tp+tn)/20000,metric['accuracy']); close(tp/(tp+fn),metric['recall'])
    close(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,metric['f1'])
freeze = json.loads((root/'models/pre-evaluation-freeze.json').read_text())
assert freeze['test_scored'] is False
for name,value in freeze['files'].items():
    assert sha(root/'models'/name) == value
for name,value in freeze['data_files'].items():
    assert sha(root/'data'/name) == value
result = {'status':'passed','architecture':platform.machine(),
          'execution_scope':'native_linuxone' if platform.machine()=='s390x' else 'operator_authorized_local_cpu',
          'rows_checked':len(rows),
          'semantic_groups':len(semantic_splits),'semantic_group_overlap':0,'paired_events_checked':len(all_ids),
          'paired_episodes':report['paired_episodes'],'runtime_confirmations':confirmed,
          'pre_fault_or_no_fault_confirmations':false_confirmations,'restored_episodes':restored,
          'held_episodes':held,'held_requests_count_as_correct':False,'raw_confusions_recomputed':True,
          'all_stream_and_endpoint_metrics_recomputed':True,'pre_evaluation_freeze_preserved':True,
          'additional_independent_trials':False,'source_commit':report['source_commit']}
with (root/'independent-audit.json').open('x') as f:
    json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
print(json.dumps(result,indent=2,sort_keys=True))
