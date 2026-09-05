"""Publication-independent static figures from saved V4 tables; no model fits."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/alphatransfer-v4-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    f=pd.read_csv(HERE/'MARKET_COMPARISON.csv');q=f[(f.scope=='KZT')&(f.track=='development_2023_2025')].set_index('config_id')
    ids=['baseline_reproduction','basis_train_120m','kzt_pooled_120m','kzt_pooled_120m__halyk_lag1','kzt_residual_shrink_120m__halyk_lag1','kzt_residual_shrink_120m__halyk_lag2']
    labels=['Исходная модель · 2 года','V3 · 10 лет','+ калибровка для KZT','+ банковские признаки Halyk','+ дообучение на KZT','Тот же подход · задержка +1 день']
    colors=['#8994a5','#64748b','#7b96b2','#61a5ba','#158b79','#d59a55']
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1.45,1]})
    y=np.arange(len(ids));v=q.loc[ids]
    a.scatter(v.brier,y,c=colors,s=100,zorder=3);a.grid(axis='x',color='#e2e8f0',linewidth=.6);a.set_yticks(y,labels);a.invert_yaxis();a.set_xlim(.16,.198);a.set_xlabel('Brier ↓');a.set_title('Качество вероятностей на одинаковых днях',loc='left',fontsize=11)
    for i,x in enumerate(v.brier):a.text(x+.00045,i,f'{x:.6f}',va='center',fontsize=9)
    b.barh(y,v.forward_delta_bps,color=colors,height=.62);b.set_yticks(y,['']*len(ids));b.invert_yaxis();b.set_xlim(0,70);b.set_xlabel('Reference advantage, bps ↑');b.set_title('Качество выбранных дней',loc='left',fontsize=11)
    for i,r in enumerate(v.itertuples()):b.text(r.forward_delta_bps+1,i,f'{r.forward_delta_bps:.1f} · n={int(r.signals)}',va='center',fontsize=9)
    fig.suptitle('Казахстан: отдельно история, калибровка, данные и дообучение',x=.04,ha='left',fontsize=15,fontweight='bold')
    fig.text(.04,-.02,'727 дат, 2023–2025 · уже исследованная история · bps по официальному курсу, не экономия клиента\nИнтервал улучшения Brier против длинной V3 включает ноль; дополнительная задержка ослабляет результат.',fontsize=9,color='#536171')
    fig.tight_layout(rect=[0,.02,1,.94]);save(fig,'01_kzt_ablation')
    s=pd.read_csv(HERE/'segments/results/headline.csv');freq=pd.read_csv(HERE/'segments/results/frequent_response_cost_sensitivity.csv')
    # Tables contain pooled fixed-weight records; require one row per displayed arm.
    policies=['v3_readiness','universal','universal_expected_budget','group_aware'];u=s[s.policy.isin(policies)].set_index('policy').loc[policies]
    fig,(a,b)=plt.subplots(1,2,figsize=(13,4.9),gridspec_kw={'width_ratios':[1,1.15]})
    bars=a.bar(range(4),u.net_scenario_value_rub,color=['#94a3b8','#158b79','#6fac99','#dc9858'],width=.65)
    a.set_xticks(range(4),['V3 readiness','Общая\nнастройка','Общая при\nexpected budget','Отдельно\nпо группам']);a.set_ylabel('Сценарная net value, RUB/клиент-период');a.set_ylim(0,750);a.set_title('Взвешенный итог всех групп',loc='left',fontsize=11)
    for bar,value in zip(bars,u.net_scenario_value_rub):a.text(bar.get_x()+bar.get_width()/2,value+12,f'{value:.1f}',ha='center')
    scenarios=['zero_response','low_response','base','high_response'];styles={'universal':('#158b79','Общая настройка'),'group_aware':('#dc9858','Частая группа')}
    for policy,(color,label) in styles.items():
        values=freq[freq.policy.eq(policy)].set_index('scenario').loc[scenarios,'net_scenario_value_rub'];b.plot([0,10,35,70],values,'o-',lw=2,color=color,label=label)
    b.axhline(0,color='#cbd5e1',lw=1);b.set_xlabel('Заданный response, %');b.set_ylabel('Сценарная net value, RUB/клиент-период');b.set_title('Frequent: результат зависит от отклика',loc='left',fontsize=11);b.legend(frameon=False)
    fig.suptitle('Персонализация: полезный частный tradeoff, отрицательный общий итог',x=.04,ha='left',fontsize=14,fontweight='bold')
    fig.text(.04,-.02,'2024–2025 · одинаковые синтетические обязательства и market paths · веса и response — сценарий, не оценка клиентов Альфы\nЧисло синтетических клиентов не увеличивает число независимых рыночных дней.',fontsize=9,color='#536171')
    fig.tight_layout(rect=[0,.03,1,.92]);save(fig,'02_segment_tradeoff')
if __name__=='__main__':main()
