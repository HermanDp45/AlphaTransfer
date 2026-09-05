"""Standalone research figure: conditional month-block intervals, no smoothing."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/'paired_intervals.csv').open()))
names=['oxr_basis_120m_delay24h','oxr_basis_120m_since2020','oxr_basis_120m_since2022']
fig,ax=plt.subplots(figsize=(9,4.9),layout='constrained')
colors=['#20628F','#CA633B']
tracks=['development_2023_2025','test2026_january_freeze']
labels=['2023–2025: разработка','2026: январская фиксация']
for j,(track,color,label) in enumerate(zip(tracks,colors,labels)):
    selected=[next(r for r in rows if r['track']==track and r['scope']=='all' and r['block']=='month' and r['config_id']==name) for name in names]
    y=[1000*float(r['delta_brier']) for r in selected]
    lo=[1000*(float(r['delta_brier'])-float(r['ci_low'])) for r in selected]
    hi=[1000*(float(r['ci_high'])-float(r['delta_brier'])) for r in selected]
    x=[i+(-.10 if j==0 else .10) for i in range(3)]
    ax.errorbar(x,y,yerr=[lo,hi],fmt='o',markersize=7,capsize=4,color=color,label=label)
ax.axhline(0,color='#344250',linewidth=1)
ax.set_xticks([0,1,2],['С июня 2018','С января 2020','С января 2022'])
ax.set_xlabel('Начало доступной истории OXR; train V3 всегда 10 лет',labelpad=10)
ax.set_ylabel('ΔBrier относительно V3 × 1000\nОтрицательное значение — лучше')
ax.set_title('Более ранняя история OXR не даёт устойчивого прироста',loc='left',fontsize=14,pad=14)
ax.grid(axis='y',alpha=.16);ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower left',frameon=False,fontsize=9)
fig.get_layout_engine().set(rect=(0,.10,1,.90))
fig.text(.02,.025,'95% интервалы: 10 000 парных месячных блоков. Условны на моделях; перебор вариантов не учтён.\nОдна и та же группа basis-признаков. Контроли глубины 2020/2022 — дополнительная ретроспективная проверка.',fontsize=8.5,color='#555')
fig.savefig(HERE/'history_depth.png',dpi=180)
fig.savefig(HERE/'history_depth.svg')
