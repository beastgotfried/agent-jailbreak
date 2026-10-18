"""Dark-theme presentation visuals for the Veyra web dashboard.

All values come from the frozen native studies. Backgrounds are transparent so
the art sits on the Carbon gray-100 page surface. Same data as the light
pitch visuals; no new measurements are introduced here.
"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FONTS = Path(__file__).resolve().parent / 'public/fonts'
for p in FONTS.glob('*.ttf'):
    font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    'font.family': 'IBM Plex Sans', 'font.size': 13,
    'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.facecolor': 'none',
    'savefig.transparent': True, 'text.color': '#f4f4f4', 'axes.labelcolor': '#a8a8a8',
    'xtick.color': '#a8a8a8', 'ytick.color': '#a8a8a8', 'svg.fonttype': 'path'})

ACC, TEAL, INK, MUT, GRID, DANGER = '#78a9ff', '#3ddbd9', '#f4f4f4', '#a8a8a8', '#393939', '#ff8389'
OUT = Path(__file__).resolve().parent / 'public/visuals'
metrics = json.loads((ROOT / 'presentation/evidence/metrics.json').read_text())

def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# 1. Runtime recovery trace (dark) -----------------------------------------
rows = list(csv.DictReader((ROOT / 'presentation/tables/scale-timeline.csv').open()))
x = [int(r['index']) for r in rows]
a = [float(r['A_absolute_baseline_error']) for r in rows]
b = [float(r['B_absolute_baseline_error']) for r in rows]
fig, ax = plt.subplots(figsize=(11.4, 4.1))
fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.87)
ax.axvspan(128, 175, color=ACC, alpha=.07)
ax.plot(x, a, color=MUT, lw=1.1, label='Guardrail only', alpha=.9)
ax.plot(x, b, color=ACC, lw=1.8, label='Veyra')
for idx, col in [(128, DANGER), (143, TEAL), (175, ACC)]:
    ax.axvline(idx, color=col, lw=1.1, ls='--', alpha=.9)
ax.text(124, .83, '128 fault ', color=DANGER, fontsize=12, ha='right', va='top')
ax.text(143, .62, ' 143 confirmed', color=TEAL, fontsize=12, va='top')
ax.text(179, .83, '175 verified recovery', color=ACC, fontsize=12, ha='left', va='top')
ax.set_xlim(0, 511); ax.set_ylim(-.03, .86)
ax.set_xlabel('Request index', labelpad=8); ax.set_ylabel('Score error vs baseline', labelpad=8)
ax.grid(axis='y', color=GRID, lw=.6); ax.spines[['top', 'right', 'left']].set_visible(False)
ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=12, labelcolor=INK)
save(fig, 'runtime-trace')

# 2. Complete-stream outcomes (dark) ----------------------------------------
A, B = metrics['overall']['A'], metrics['overall']['B']
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.1))
fig.subplots_adjust(left=.02, right=.99, bottom=.2, top=.8, wspace=.32)
panels = [
    ('Correct automatic decisions', 100*A['operational_accuracy'], 100*B['operational_accuracy'], '%', 100),
    ('Harmful allow rate', 100*A['harmful_allow_rate'], 100*B['harmful_allow_rate'], '%', 40),
    ('Safe refusal rate', 100*A['safe_refusal_rate'], 100*B['safe_refusal_rate'], '%', 40)]
for ax, (title, va, vb, unit, lim) in zip(axes, panels):
    ax.barh([1, 0], [va, vb], height=.5, color=[GRID, ACC])
    ax.set_xlim(0, lim*1.28); ax.set_ylim(-.62, 1.62)
    ax.set_yticks([1, 0], ['Guardrail only', 'Veyra'], fontsize=11.5)
    ax.set_title(title, fontsize=14.5, color=INK, loc='left', pad=14, fontweight='semibold')
    for y, v in [(1, va), (0, vb)]:
        ax.text(v + lim*.025, y, f'{v:.2f}{unit}', va='center', fontsize=13.5,
                color=ACC if y == 0 else MUT, fontweight='semibold')
    ax.set_xticks([]); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
save(fig, 'stream-outcomes')

# 3. Response outcomes stacked bar (dark) ------------------------------------
fig, ax = plt.subplots(figsize=(12.4, 1.9))
fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
left = 0
for value, color, label, tc in [(45, ACC, '45 restored', '#001141'), (5, TEAL, '5 held', '#001141'),
                                (10, GRID, '10 controls', INK)]:
    ax.barh(.5, value, left=left, height=.62, color=color)
    ax.text(left + value/2, .5, label, ha='center', va='center', fontsize=13,
            color=tc, fontweight='semibold')
    left += value
ax.set_xlim(0, 60); ax.set_ylim(0, 1); ax.axis('off')
save(fig, 'response-outcomes')

# 4. Data split (dark) --------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.4, 1.7))
fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
left = 0
for value, color, label, tc in [(70, ACC, '70,000 train', '#001141'), (10, TEAL, '10,000\nvalidation', '#001141'),
                                (20, GRID, '20,000 test', INK)]:
    ax.barh(.5, value, left=left, height=.72, color=color)
    ax.text(left + value/2, .5, label, ha='center', va='center', fontsize=12.5 if value == 10 else 13.5,
            color=tc, fontweight='semibold', linespacing=1.2)
    left += value
ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.axis('off')
save(fig, 'data-split')

# 5. Healthy classifier profile (dark) ----------------------------------------
g = metrics['healthy_test']['guardrail']
fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.6))
fig.subplots_adjust(left=.02, right=.99, bottom=.04, top=.78, wspace=.25)
stats = [('Accuracy', g['accuracy']), ('Harmful recall', g['recall']),
         ('Safe false-refusal rate', g['fpr']), ('ROC-AUC', g['roc_auc'])]
for ax, (title, v) in zip(axes, stats):
    ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, .78), 1, .03, color=GRID))
    ax.add_patch(plt.Rectangle((0, .78), v, .03, color=ACC))
    ax.text(0, .5, title, fontsize=12.5, color=MUT)
    ax.text(0, .06, f'{v*100:.2f}%' if title != 'ROC-AUC' else f'{v:.4f}',
            fontsize=27, color=INK, fontweight='semibold')
save(fig, 'classifier-profile')

print('dark visuals written to', OUT)
