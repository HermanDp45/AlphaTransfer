#!/usr/bin/env python3
"""Aggregate World Bank L2KGZ data locally. Never export household/person rows.
WB reference KGZ_2021-2025_L2KGZ_v02_M; DOI10.48529/swmc-aq82.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,default=Path('/private/tmp/at-kg-microdata.zip'));a=ap.parse_args();out=HERE/'data';out.mkdir(exist_ok=True)
 with zipfile.ZipFile(a.zip) as z:
  h=pd.read_csv(z.open('l2kgz_cati_household_data_42.csv'),usecols=['round','hhid','date','popw','hhsize'])
  i=pd.read_csv(z.open('l2kgz_cati_individual_data_42.csv'),usecols=['round','hhid','fmid','mig_living_remittance','mig_living_remittance_method','mig_living_country','remittance_hh'])
 assert not h.duplicated(['round','hhid']).any() and not i.duplicated(['round','hhid','fmid']).any()
 h['date']=pd.to_datetime(h.date);h['month']=h.date.dt.to_period('M').astype(str);roundmonth=h.groupby('round').month.agg(lambda x:x.mode().iloc[0]);h['wave_month']=h['round'].map(roundmonth)
 h['household_weight_derived']=h.popw/h.hhsize
 for c in ('mig_living_remittance','mig_living_remittance_method','mig_living_country','remittance_hh'):i[c]=i[c].str.strip()
 i=i.merge(h[['round','hhid','wave_month','popw','household_weight_derived']],on=['round','hhid'],validate='many_to_one')
 m=i[i.mig_living_remittance.isin(['Yes','No'])].copy();m['sent']=m.mig_living_remittance.eq('Yes').astype(int)
 monthly=[]
 for wave,g in m.groupby('wave_month'):
  monthly.append(dict(wave_month=wave,observed_migrant_records=len(g),reporting_migrants=g.sent.sum(),unweighted_sent_fraction=g.sent.mean(),household_weighted_sent_fraction=np.average(g.sent,weights=g.household_weight_derived)))
 pd.DataFrame(monthly).to_csv(out/'l2kgz_migrant_month_aggregates.csv',index=False)
 bins=[];transitions=[];methods=[]
 for label,lo,hi in [('pre2024_observations_released_later','2021-12','2023-12'),('full2021_2025','2021-12','2025-12')]:
  sub=m[m.wave_month.between(lo,hi)].sort_values(['hhid','fmid','wave_month'])
  g=sub.groupby(['hhid','fmid']).agg(months=('sent','size'),sent_months=('sent','sum'),weight=('household_weight_derived','mean'))
  for minimum in (6,12):
   eligible=g[(g.months>=minimum)&(g.sent_months>0)].copy();eligible['rate']=eligible.sent_months/eligible.months
   eligible['band']=pd.cut(eligible.rate,[0,.25,.75,1],labels=['sporadic_le25pct_months','intermittent_25to75pct_months','regular_gt75pct_months'],include_lowest=True)
   for band,b in eligible.groupby('band',observed=False):
    bins.append(dict(period=label,min_observed_months=minimum,band=str(band),migrants=len(b),eligible_migrants=len(eligible),unweighted_share=len(b)/len(eligible),derived_household_weighted_share=b.weight.sum()/eligible.weight.sum(),mean_reporting_fraction=b.rate.mean()))
  # Consecutive calendar waves only, conditional on same migrant being observable in both.
  sub['month_ordinal']=pd.PeriodIndex(sub.wave_month,freq='M').asi8
  prior=sub.groupby(['hhid','fmid'])[['month_ordinal','sent']].shift(1)
  pairs=sub[sub.month_ordinal.sub(prior.month_ordinal).eq(1)].copy();pairs['previous_sent']=prior.loc[pairs.index,'sent']
  for previous,part in pairs.groupby('previous_sent'):
   transitions.append(dict(period=label,previous_sent=int(previous),consecutive_pairs=len(part),current_sent=part.sent.sum(),unweighted_p_next_sent=part.sent.mean(),derived_household_weighted_p_next_sent=np.average(part.sent,weights=part.household_weight_derived)))
  sent=sub[sub.sent.eq(1)]
  for method,b in sent.groupby('mig_living_remittance_method'):
   methods.append(dict(period=label,method=method,migrant_months=len(b),weighted_share=b.household_weight_derived.sum()/sent.household_weight_derived.sum()))
 pd.DataFrame(bins).to_csv(out/'l2kgz_regularity_bands.csv',index=False);pd.DataFrame(transitions).to_csv(out/'l2kgz_transitions.csv',index=False);pd.DataFrame(methods).to_csv(out/'l2kgz_transfer_methods.csv',index=False)
 report={'reference_id':'KGZ_2021-2025_L2KGZ_v02_M','doi':'https://doi.org/10.48529/swmc-aq82','catalog':'https://microdata.worldbank.org/catalog/6523','download':'https://microdata.worldbank.org/catalog/6523/download/329993','sha256':hashlib.sha256(a.zip.read_bytes()).hexdigest(),'zip_bytes':a.zip.stat().st_size,'household_month_rows':len(h),'individual_month_rows':len(i),'observed_migrant_months':len(m),'observed_migrants':m.groupby(['hhid','fmid']).ngroups,'wave_months':len(roundmonth),'raw_microdata_redistributed':False,'weighting':'popw is population weight; popw/hhsize derived household weight, then average across observed waves for migrant bands. No official longitudinal migrant weight is provided: weighted bands are descriptive, not design-consistent population estimates.','known_at':'Current v02 archive released after many observation dates; pre2024 subset is not a point-in-time available release. Used as scenario evidence only, not a historic bank covariate.','limitations':['One yes/no per migrant-month cannot identify within-month transfer frequency.','Questions refer to sending to Kyrgyz recipient household, not Alfa app users.','Missing migrant questions are unobserved, not no transfer.','Panel persistence conditions on continued observation and >=6/12 observed months; attrition and migrant composition affect results.','Month assigned by modal calendar month of survey round; April2024 gap is excluded from consecutive transitions.','No household or member IDs or raw rows exported.','No PSU/strata design variance estimated; weighting not formal population CI.']}
 (out/'l2kgz_aggregate_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(pd.DataFrame(bins).to_string(index=False));print(pd.DataFrame(transitions).to_string(index=False))
if __name__=='__main__':main()
