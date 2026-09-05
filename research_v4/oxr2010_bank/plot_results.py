from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def read(name):
 with (HERE/name).open() as f:return list(csv.DictReader(f))
def main():
 ci=read('long_models/paired_intervals.csv');sim=read('long_models/simultaneous_intervals.csv')
 def point(name):return next(x for x in ci if x['track']=='development_2023_2025' and x['contrast']==name and x['block']=='month')
 fig,axes=plt.subplots(1,2,figsize=(13,5.5));fig.patch.set_facecolor('#f8fafc')
 names=[('history_primary','OXR2010 vs2018\nV3, 120m'),('history_longer','OXR2010 vs2018\nV3, до180m'),('history_bank_primary','OXR2010 vs2018\nHalyk + KZT'),('treasury_history','OXR2010 vs2018\nV3 + Treasury')]
 for i,(key,label) in enumerate(names):
  r=point(key);d=float(r['delta_brier']);lo=float(r['ci_low']);hi=float(r['ci_high'])
  axes[0].errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=5,color='#ad4728')
 axes[0].set_yticks(range(4),[x[1] for x in names]);axes[0].invert_yaxis();axes[0].set_title('Добавление истории OXR до2018\nПарные95% интервалы',pad=18)
 keys=[('bank_vs_v3','Halyk + KZT\nvs V3 KZT'),('treasury_plus_bank_vs_v3','Treasury + Halyk + KZT\nvs V3 KZT')]
 for i,(key,label) in enumerate(keys):
  r=point(key);d=float(r['delta_brier']);lo=float(r['ci_low']);hi=float(r['ci_high'])
  axes[1].errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=5,color='#087f8c',label='Парный интервал' if i==0 else None)
 s=next(x for x in sim if x['track']=='development_2023_2025' and x['config_id']=='treasury_halyk_shrink_120m' and x['block']=='month')
 d=float(s['delta_brier']);lo=float(s['simultaneous_ci_low']);hi=float(s['simultaneous_ci_high'])
 axes[1].errorbar(d,2,xerr=[[d-lo],[hi-d]],fmt='s',capsize=5,color='#7140a5',label='Поправка на27 моделей')
 axes[1].set_yticks(range(3),[x[1] for x in keys]+['Тот же кандидат\nс поправкой на поиск']);axes[1].set_ylim(2.6,-.6)
 axes[1].set_title('Банковская специализация\nЛучший кандидат и пределы доказательства',pad=18)
 for ax in axes:
  ax.axvline(0,color='#667085',ls='--',lw=1);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Δ Brier: левее нуля — лучше');ax.spines[['top','right','left']].set_visible(False)
 fig.suptitle('2023–2025 · одинаковые даты · меньший Brier лучше',fontsize=14,y=.99)
 fig.text(.5,.02,'Ретроспективное исследование. Brier не равен экономии клиента. Старые данные OXR не дали устойчивого выигрыша.',ha='center',fontsize=10,color='#475467')
 fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(HERE/'results.png',dpi=160,facecolor=fig.get_facecolor());fig.savefig(HERE/'results.svg',facecolor=fig.get_facecolor())
if __name__=='__main__':main()
