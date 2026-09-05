"""Export publication-quality research figures from saved metric artifacts."""
import csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/alphatransfer-plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
def read(path):
 with path.open() as f:return list(csv.DictReader(f))
rows=read(ROOT/'models/summary_h5.csv')
rows={r['config_id']:r for r in rows if r['track']=='development_2023_2025'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlepad':14,'savefig.facecolor':'white'})
blue='#195b83';teal='#07877b';red='#c54936';gray='#7f8991'
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
 plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12.7,4.8),layout='constrained')
labels=['Existing HGB + CNY','10-year history','HGB + TabM blend','10-year + Treasury']
tabm=next(r for r in read(ROOT/'tabm/output/aggregate_metrics.csv') if r['config_id']=='blend_plus_cnyrub_basis' and r['period']=='development')
combo=next(r for r in read(ROOT/'external_data/long_combo/development_metrics.csv') if r['config_id']=='basis_train_120m__treasury_lag7')
brier=[float(rows['baseline_reproduction']['brier']),float(rows['basis_train_120m']['brier']),float(tabm['model_brier']),float(combo['brier'])]
lift=[float(rows['baseline_reproduction']['lift']),float(rows['basis_train_120m']['lift']),float(tabm['candidate_cell_standardized_lift']),float(combo['candidate_cell_standardized_lift'])]
for ax,values,title in [(axs[0],brier,'Probability error (Brier) — lower is better'),(axs[1],lift,'Signal hit-rate lift — higher is better')]:
 ax.scatter(values,labels,c=[blue,teal,teal,teal],s=90,zorder=3);ax.axvline(values[0],color=gray,ls='--',alpha=.5);ax.invert_yaxis();ax.set_title(title)
 ax.set_xlim((.17,.199) if ax==axs[0] else (1.1,1.50))
 for i,v in enumerate(values):ax.text(v+.0004 if ax==axs[0] else v+.007,i,f'{v:.4f}' if ax==axs[0] else f'{v:.3f}',va='center')
 ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Better forecasts did not improve the existing notification policy',fontsize=15)
fig.supxlabel('2023–2025 exploratory OOT • Same 727 dates / five corridors • No confirmed overall replacement',fontsize=10,color=gray)
save(fig,'01_probability_vs_policy')
fig,ax=plt.subplots(figsize=(10.5,5.2),layout='constrained')
xs=[3,6,12,24,36,60,120]
names=[f'basis_train_{m}m' if m!=24 else 'baseline_reproduction' for m in xs]
y=[float(rows[n]['brier']) for n in names]
ax.plot(xs,y,'o-',color=blue,lw=2.4,label='Core FX + CNY basis')
ax.axhline(y[3],color=red,ls='--',lw=1.3,label='Existing two-year model')
ax.set_xscale('log',base=2);ax.set_xticks(xs,[str(x) for x in xs]);ax.set_xlabel('Training history, months (12-month calibration interval separate)');ax.set_ylabel('Brier score');ax.set_title('More history helps non-monotonically')
ax.grid(alpha=.18);ax.legend(frameon=False)
fig.supxlabel('CBR history extended to 2010 • CNY basis missing before 2020 in this ladder • 95% paired CI includes zero',fontsize=10,color=gray)
save(fig,'02_history_curve')
fr=read(ROOT/'models/risk_coverage_frontier.csv')
r=[r for r in fr if r['config_id']=='baseline_reproduction' and r['track']=='development_2023_2025' and r['policy'].startswith('fixed_probability')]
fig,ax=plt.subplots(figsize=(10.5,5.2),layout='constrained')
ax.plot([float(x['mean_per_corridor_week']) for x in r],[float(x['forward_delta_bps']) for x in r],'-o',color=teal)
for x in r:
 if float(x['threshold']) in [0,.3,.5,.65]:
  ax.annotate('p ≥ '+x['threshold'],(float(x['mean_per_corridor_week']),float(x['forward_delta_bps'])),xytext=(6,7),textcoords='offset points',fontsize=9)
ax.scatter([.8864],[47.6856],marker='*',s=200,color=red,label='Existing validation / cadence policy',zorder=4)
ax.set_xlabel('Candidate signals per corridor per week');ax.set_ylabel('Forward official-reference advantage, bps');ax.set_title('A scarce communication slot changes the useful operating point');ax.grid(alpha=.18);ax.legend(frameon=False)
fig.supxlabel('Diagnostic risk–coverage frontier, not threshold selection on a fresh holdout • Reference rates, not bank savings',fontsize=10,color=gray)
save(fig,'03_risk_coverage')
h=json.loads((ROOT/'behavior/results/headline_metrics.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
fields=[('Contacts per client-period','market_only_contacts_per_client_period','user_aware_contacts_per_client_period'),('Relevant contacts, %','market_only_relevance_pct','user_aware_relevance_pct'),('Timing proxy, RUB / client','market_only_gross_rub_per_client_period','user_aware_gross_rub_per_client_period')]
for ax,(title,a,b) in zip(axs,fields):
 vals=[h[a],h[b]];ax.bar(['Market only','User aware'],vals,color=[blue,teal]);ax.set_title(title,fontsize=12);ax.set_ylim(0,max(vals)*1.22)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.035,f'{v:.1f}',ha='center')
 ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Personalization preserves 82% of simulated value with 84% fewer contacts',fontsize=15)
fig.supxlabel('SCENARIO SIMULATION • 3 seeds, balanced frequency segments • No observed customer uplift or proven bank realism',fontsize=10,color=red)
save(fig,'04_personalization_scenario')
print('Exported',len(list(OUT.glob('*.png'))),'PNG/SVG figures')
