from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FONTS = ROOT / 'assets' if (ROOT / 'assets/IBMPlexSans-Regular.ttf').exists() else ROOT.parents[1] / 'prd/fonts'
for path in FONTS.glob('*.ttf'):
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({'font.family': 'IBM Plex Sans', 'font.size': 16,
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'text.color': '#161616',
    'axes.labelcolor': '#393939', 'xtick.color': '#525252', 'ytick.color': '#525252',
    'svg.fonttype': 'none', 'savefig.facecolor': '#ffffff'})
BLUE, TEAL, INK, GRAY, PALE = '#0f62fe', '#007d79', '#161616', '#8d8d8d', '#edf5ff'
metrics = json.loads((ROOT / 'evidence/metrics.json').read_text())

def save(fig, name):
    fig.savefig(ROOT / 'visuals' / (name + '.png'), dpi=160)
    fig.savefig(ROOT / 'visuals' / (name + '.svg'))
    plt.close(fig)

def page(title, subtitle):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .925, title, fontsize=29, weight='semibold', va='center')
    fig.text(.055, .875, subtitle, fontsize=14, color='#525252', va='center')
    return fig

def box(ax, x, y, w, h, title, detail, fill='white', edge='#a6c8ff', size=16):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                          linewidth=1.5, edgecolor=edge, facecolor=fill, transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h*.66, title, transform=ax.transAxes, ha='center', va='center', fontsize=size, weight='semibold')
    ax.text(x + w/2, y + h*.29, detail, transform=ax.transAxes, ha='center', va='center', fontsize=11.5, color='#393939', linespacing=1.4)

def arrow(ax, start, end, color=BLUE, style='-', width=1.8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
        linewidth=width, color=color, linestyle=style, transform=ax.transAxes))

fig = page('Veyra: guardrail integrity, executed natively.', 'Implemented architecture · Selected runtime plus a separate traffic-study path')
ax = fig.add_axes([.025, .065, .95, .76]); ax.axis('off')
ax.add_patch(FancyBboxPatch((.005, .01), .99, .98, boxstyle='round,pad=0.004,rounding_size=.018',
                          facecolor=PALE, edgecolor=BLUE, linewidth=1.8, transform=ax.transAxes))
ax.text(.03, .945, 'IBM LINUXONE  /  LINUX s390x  /  CPU ONLY', transform=ax.transAxes, fontsize=16, color=BLUE, weight='semibold')
ax.text(.97, .945, 'Offline operator approval', transform=ax.transAxes, fontsize=13, color='#393939', ha='right')
ax.text(.5, .855, 'Veyra: frozen models, policy, and disjoint labelled probes', transform=ax.transAxes,
        ha='center', fontsize=13, bbox={'facecolor':'white', 'edgecolor':'#a6c8ff', 'pad':9})
xs = [.035, .275, .515, .755]; w=.205
box(ax,xs[0],.64,w,.135,'Ordered requests','CLI / Python replay')
box(ax,xs[1],.64,w,.135,'Supervisor','Gate + active backend')
box(ax,xs[2],.64,w,.135,'Guardrail + canary','Actual model inference')
box(ax,xs[3],.64,w,.135,'Frozen policy','Guardrail decision / agent stub')
for a,b in zip(xs,xs[1:]): arrow(ax,(a+w+.006,.707),(b-.008,.707))
arrow(ax,(.62,.832),(.62,.783),style='--')
box(ax,xs[0],.365,w,.14,'RuntimeWatch','8 matched probes\nevery 16 requests')
box(ax,xs[1],.365,w,.14,'RecoveryManager','Affine candidate /\ntrusted backend reload')
box(ax,xs[2],.365,w,.14,'Verify + activate','Both models + fresh check\nApproval + single-use plan')
box(ax,xs[3],.365,w,.14,'Explicit hold','No verified replacement\nFull check before release',edge=TEAL,fill='#d9fbfb')
for a,b in zip(xs,xs[1:]): arrow(ax,(a+w+.006,.434),(b-.008,.434),color=TEAL)
arrow(ax,(.62,.51),(.62,.63),color=TEAL)
ax.text(.635,.566,'Verified switch',transform=ax.transAxes,fontsize=10,color=TEAL,ha='left')
arrow(ax,(.139,.51),(.139,.565),color=BLUE,style='--')
arrow(ax,(.139,.565),(.56,.565),color=BLUE,style='--')
arrow(ax,(.56,.565),(.56,.63),color=BLUE,style='--')
ax.text(.31,.582,'Matched probe execution',transform=ax.transAxes,fontsize=10,color=BLUE,ha='center')
box(ax,.035,.12,.445,.14,'Separate traffic study','Source-bound score events → signed gap → KS warning\nWarning has no response authority',fill='#f4f4f4',edge='#c6c6c6',size=15)
box(ax,.515,.12,.445,.14,'Local evidence','JSON / JSONL events, receipts and SHA-256 manifests\nIn-memory runtime state; no broker or database service',fill='#f4f4f4',edge='#c6c6c6',size=15)
arrow(ax,(.62,.355),(.62,.27),color=GRAY,style='--')
ax.text(.035,.055,'Current interfaces: CLI and Python API. Enterprise web console and connectors remain roadmap items.',
        transform=ax.transAxes,fontsize=11,color='#525252')
fig.text(.055,.025,'Source: release 8a86ce0 · SupervisedRuntime / RuntimeWatch / RecoveryManager / DisagreementMonitor',fontsize=10,color='#6f6f6f')
save(fig,'architecture')

fig = page('Measured complete-stream effect', 'Same model and requests · 60 paired episodes · 30,720 synthetic request observations')
rows=[('Correct automatic decisions','All 30,720 observations',100*metrics['overall']['A']['operational_accuracy'],100*metrics['overall']['B']['operational_accuracy'],100),
      ('Harmful allow rate','15,840 harmful-labelled observations',100*metrics['overall']['A']['harmful_allow_rate'],100*metrics['overall']['B']['harmful_allow_rate'],40),
      ('Safe refusal rate','14,880 safe-labelled observations',100*metrics['overall']['A']['safe_refusal_rate'],100*metrics['overall']['B']['safe_refusal_rate'],40),
      ('Hold rate','All 30,720 observations',0,1685/30720*100,15)]
positions=[(.13,.51,.34,.25),(.61,.51,.32,.25),(.13,.17,.34,.25),(.61,.17,.32,.25)]
for (title,unit,a,b,limit),pos in zip(rows,positions):
    ax=fig.add_axes(pos)
    ax.barh([1,0],[a,b],height=.45,color=[GRAY,BLUE])
    ax.set_yticks([1,0],['Guardrail only','Monitored'],fontsize=12)
    ax.set_xlim(0,limit*1.14); ax.set_ylim(-.6,1.8)
    ax.text(0,1.67,title,fontsize=19,weight='semibold')
    ax.text(0,1.37,unit,fontsize=11,color='#6f6f6f')
    for y,v in [(1,a),(0,b)]: ax.text(v+limit*.02,y,f'{v:.2f}%',va='center',fontsize=16,weight='semibold')
    ax.set_xticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(axis='y',length=0)
fig.text(.055,.065,'Holds count as zero correct automatic decisions. More containment is not the same as restored service.',fontsize=12,color='#525252')
fig.text(.055,.03,'Source: synthetic-100k-20261017-01 / benchmark-report.json / overall. Timed and labelled scope remains experimental.',fontsize=10,color='#6f6f6f')
save(fig,'stream-outcomes')

fig=page('From a changed output to a verified recovery', 'Actual native scale / seed-0 trace · Absolute guardrail score error against the archived healthy score')
with (ROOT/'tables/scale-timeline.csv').open() as f: trace=list(csv.DictReader(f))
x=[int(r['index']) for r in trace]
a=[float(r['A_absolute_baseline_error']) for r in trace]
b=[float(r['B_absolute_baseline_error']) for r in trace]
ax=fig.add_axes([.075,.23,.865,.52])
ax.axvspan(128,175,color='#edf5ff',alpha=1)
ax.plot(x,a,color=GRAY,lw=1.6,label='Guardrail only',alpha=.85)
ax.plot(x,b,color=BLUE,lw=2,label='Monitored + verified response')
for idx,color in [(128,'#525252'),(143,TEAL),(175,BLUE)]:ax.axvline(idx,color=color,lw=1.3,linestyle='--')
ax.set_xlim(0,511);ax.set_ylim(-.025,.84);ax.set_xlabel('Request index (zero-based)',labelpad=12);ax.set_ylabel('Absolute score error',labelpad=10)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#e0e0e0',lw=.7)
ax.legend(frameon=False,loc='upper right',fontsize=13)
fig.text(.08,.79,'128  Fault starts',fontsize=15,color='#525252',weight='semibold')
fig.text(.36,.79,'143  Runtime confirms',fontsize=15,color=TEAL,weight='semibold')
fig.text(.66,.79,'175  Verified correction',fontsize=15,color=BLUE,weight='semibold')
fig.text(.075,.115,'15-request confirmation delay  +  32-position response delay',fontsize=17,weight='semibold')
fig.text(.075,.075,'Endpoint: 1,000 requests · Guardrail error 0.0 · Canary error 0.0 · Decision mismatches 0',fontsize=13,color='#525252')
fig.text(.075,.03,'Recorded native replay, not a live dashboard. No smoothing. Full-stream outcomes include the response delay.',fontsize=10,color='#6f6f6f')
save(fig,'runtime-recovery')

fig=page('Confirm. Restore. Contain.', 'Selected native runtime experiment · 50 fault episodes and 10 no-fault controls')
ax=fig.add_axes([.06,.37,.88,.28]);ax.axis('off');ax.set_xlim(0,60);ax.set_ylim(0,1)
left=0
for value,color,title in [(45,BLUE,'45\nRestored'),(5,TEAL,'5\nHeld'),(10,'#e0e0e0','10\nControls')]:
    ax.barh(.5,value,left=left,height=.54,color=color)
    ax.text(left+value/2,.5,title,ha='center',va='center',fontsize=22 if value>5 else 17,
            color='white' if color!= '#e0e0e0' else INK,weight='semibold')
    left+=value
fig.text(.085,.735,'55 / 55',fontsize=47,weight='semibold',color=BLUE)
fig.text(.085,.685,'Injected runtime incidents confirmed',fontsize=15)
fig.text(.55,.735,'0 / 10',fontsize=47,weight='semibold',color=TEAL)
fig.text(.55,.685,'No-fault episodes with false confirmations',fontsize=15)
fig.text(.075,.265,'15 request positions',fontsize=23,weight='semibold')
fig.text(.075,.22,'Confirmation delay at the tested fault onsets',fontsize=14,color='#525252')
fig.text(.075,.13,'Five recurrent-fault episodes contain two incidents each. Holds remain distinct from restored service.',fontsize=13,color='#525252')
fig.text(.075,.085,'Finite controlled suite. These results do not estimate a production-wide detection or false-alarm rate.',fontsize=12,color='#525252')
fig.text(.075,.03,'Source: synthetic-100k-20261017-01 / independent-audit.json and benchmark-report.json',fontsize=10,color='#6f6f6f')
save(fig,'response-outcomes')

fig=page('Split related intents before expansion', '100,000 compositional synthetic requests · Class-balanced train, validation and test partitions')
ax=fig.add_axes([.065,.49,.87,.22]);ax.axis('off');ax.set_xlim(0,100);ax.set_ylim(0,1)
left=0
for val,color,label in [(70,BLUE,'70,000\nTrain'),(10,TEAL,'10,000\nValidation'),(20,'#a6c8ff','20,000\nTest')]:
    ax.barh(.5,val,left=left,height=.7,color=color)
    ax.text(left+val/2,.5,label,ha='center',va='center',fontsize=21 if val!=10 else 14,
            weight='semibold',color=INK if val==20 else 'white')
    left+=val
for px,number,label in [(.08,'100','Template families'),(.38,'91','Reviewed semantic groups'),(.69,'0','Cross-split group overlap')]:
    fig.text(px,.34,number,fontsize=45,weight='semibold',color=BLUE)
    fig.text(px,.285,label,fontsize=16)
fig.text(.075,.19,'Only text enters the classifiers. Matched labels, contexts, and frames remain in the same partition.',fontsize=15,color='#393939')
fig.text(.075,.13,'Validation roles: 5,000 threshold / 3,000 reserved calibration / 1,000 probe fit / 1,000 probe verification.',fontsize=13,color='#525252')
fig.text(.075,.075,'Variants are not independent human requests. Reviewed grouping does not prove complete semantic independence.',fontsize=12,color='#525252')
fig.text(.075,.03,'Source: data-manifest.json and src/data/synthetic.py · Selected frozen synthetic study',fontsize=10,color='#6f6f6f')
save(fig,'data-split')

fig=page('Native. Offline. Reproducible.', 'The full selected workflow executes on IBM LinuxONE s390x CPUs')
ax=fig.add_axes([.045,.33,.91,.4]);ax.axis('off')
labels=[('Generate','100,000 rows'),('Fit','39.82 seconds'),('Infer','30,720 observations'),('Detect','55 / 55 incidents'),('Respond','45 restored / 5 held'),('Audit','Independent checks')]
width=.145
for i,(title,detail) in enumerate(labels):
    x=.005+i*.169
    box(ax,x,.34,width,.40,title,detail,fill=PALE,edge='#a6c8ff',size=17)
    if i<5:arrow(ax,(x+width+.004,.54),(x+.165,.54))
fig.text(.075,.735,'Linux s390x  ·  Big-endian  ·  CPU-only  ·  Two-CPU quota / 4 GiB container limit',fontsize=18,color=BLUE,weight='semibold')
fig.text(.075,.25,'292 native tests pass',fontsize=26,weight='semibold')
fig.text(.57,.25,'Fresh offline container',fontsize=26,weight='semibold')
fig.text(.075,.20,'Four optional PyArrow converter tests skip.',fontsize=13,color='#525252')
fig.text(.57,.20,'Data, learned state, scores and decisions match.',fontsize=13,color='#525252')
fig.text(.075,.115,'Highest study child-process peak RSS: 923.84 MiB. This is an experimental resource profile, not enterprise capacity.',fontsize=13,color='#525252')
fig.text(.075,.075,'Native CPU verification does not imply Telum/Spyre acceleration, hardware attestation, or IBM certification.',fontsize=12,color='#525252')
fig.text(.075,.03,'Sources: selected study execution receipt; release-package-20261017-02 qualification and reproduction receipts',fontsize=10,color='#6f6f6f')
save(fig,'native-proof')
print('Created six PNG/SVG visual pairs.')
