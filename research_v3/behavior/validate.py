#!/usr/bin/env python3
"""Decision-invariance and boundary checks for the behavioral experiment."""
from datetime import date
import numpy as np
import pandas as pd
from simulate import Scenario,make_users,make_world,run_policy,cap_dates
from preview_gate import BehaviorContext,preview_readiness_gate

user=make_users(4101,20)[0]
assert user['last_preperiod_transfer']<0
n=366;ds=pd.date_range('2024-01-01',periods=n);world=make_world(user,2024,n,4101,Scenario())
valid=np.arange(n);candidates=cap_dates(np.arange(0,n,3))
a=run_policy('user_aware',candidates,valid,ds,np.ones(n),np.zeros(n),np.zeros(n),world,Scenario())
b=run_policy('user_aware',candidates,valid,ds,np.exp(np.arange(n)/1000),np.ones(n),np.ones(n)*10000,world,Scenario())
assert np.array_equal(a['_contact_days'],b['_contact_days']), 'future FX/labels cannot affect decision'
assert a['gross_timing_value_rub']==0, 'flat-FX path has zero gross timing value'
assert a['policy_volume_rub']==a['organic_volume_rub']
assert a['completed_transfers']==a['planned_transfers']
assert a['no_delay'] and a['no_unfunded'] and a['no_urgent_shift']
assert a['cap_max']<=2
ctx=BehaviorContext(date(2026,9,4),5,date(2026,9,7),True,False,date(2026,8,8),False,True,1,date(2026,9,4))
assert preview_readiness_gate(ctx)['eligible']
ctx2=BehaviorContext(date(2026,9,4),0,None,None,False,None,False,True,1,date(2026,9,4))
assert not preview_readiness_gate(ctx2)['eligible']
try:
    preview_readiness_gate(BehaviorContext(date(2026,9,4),0,None,True,True,None,False,True,1,date(2026,9,5)))
except ValueError:pass
else:raise AssertionError('future-known context must reject')
print('PASS: no future-outcome selection; zero value on flat FX; conservation; bounds; preview fail-closed')
