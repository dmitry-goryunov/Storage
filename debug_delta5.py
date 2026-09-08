import re, numpy as np, pandas as pd
import storage_model
from storage_model import Storage

quotes = pd.read_excel('ttf q.xlsx')
quotes = quotes.rename(columns={quotes.columns[0]: 'quote_date'})
quotes = quotes.dropna(subset=['quote_date']).copy()
quotes['quote_date'] = pd.to_datetime(quotes['quote_date'], format='mixed')
quotes = quotes.sort_values('quote_date').reset_index(drop=True)
contract_columns = sorted(
    [c for c in quotes.columns if re.fullmatch(r'TTFc\d+', str(c))],
    key=lambda c: int(re.search(r'\d+', str(c)).group()),
)

def _month_start(ts): ts=pd.Timestamp(ts); return pd.Timestamp(ts.year,ts.month,1)
def _month_end(ts): return _month_start(ts)+pd.offsets.MonthEnd(0)
def _front_month_start(qd): return _month_start(qd)+pd.DateOffset(months=1)

def monthly_curve_from_quote(row):
    front_month = _front_month_start(row['quote_date'])
    data=[]
    for col in contract_columns:
        value=row[col]
        if pd.isna(value): continue
        n=int(re.search(r'\d+',col).group())
        cs=front_month+pd.DateOffset(months=n-1)
        data.append((cs,float(value),col))
    if not data: return pd.Series(dtype=float),pd.DataFrame()
    cdf=pd.DataFrame(data,columns=['contractStart','value','contract'])
    cdf['contractEnd']=cdf['contractStart'].map(_month_end)
    cdf=cdf[['contract','contractStart','contractEnd','value']]
    monthly=cdf.set_index('contractStart')['value'].sort_index().rename('value')
    return monthly,cdf

def curve_df_for_storage(row,curve_start=None,include_da=True):
    monthly,curve_df=monthly_curve_from_quote(row)
    curve_df=curve_df.copy()
    if not monthly.empty:
        full_months=pd.date_range(monthly.index.min(),monthly.index.max(),freq='MS')
        monthly=monthly.reindex(full_months).interpolate(method='time').ffill().bfill()
        curve_df=pd.DataFrame({'contract':[f'TTFc{i+1}' for i in range(len(monthly))],'contractStart':monthly.index,'value':monthly.values})
        curve_df['contractEnd']=curve_df['contractStart'].map(_month_end)
        curve_df=curve_df[['contract','contractStart','contractEnd','value']]
    qd=pd.Timestamp(row['quote_date'])
    cs=pd.Timestamp(curve_start) if curve_start is not None else qd
    if include_da and pd.notna(row.get('DA',np.nan)):
        fm=_front_month_start(qd)
        da_end=fm-pd.Timedelta(days=1)
        da_row=pd.DataFrame([{'contract':'DA','contractStart':min(cs,qd),'contractEnd':da_end,'value':float(row['DA'])}])
        curve_df=pd.concat([da_row,curve_df],ignore_index=True)
    elif cs<curve_df['contractStart'].min():
        fv=float(curve_df.sort_values('contractStart').iloc[0]['value'])
        fr=curve_df['contractStart'].min()
        stub=pd.DataFrame([{'contract':'FRONT_STUB','contractStart':cs,'contractEnd':fr-pd.Timedelta(days=1),'value':fv}])
        curve_df=pd.concat([stub,curve_df],ignore_index=True)
    return curve_df[['contractStart','contractEnd','value']].sort_values('contractStart').reset_index(drop=True)

FDDate=pd.Timestamp('2026-01-05')
valDate=pd.Timestamp('2026-01-01')
storageStart=pd.Timestamp('2027-01-01')
storageEnd=pd.Timestamp('2027-12-31')
days=30; vol=0.50; n_p_full=30; v_step=1000

eligible=quotes.loc[quotes['quote_date']<=FDDate]
eligible=eligible.loc[eligible[contract_columns].notna().any(axis=1)]
fd_quote=eligible.iloc[-1]
curve=curve_df_for_storage(fd_quote,curve_start=valDate,include_da=True)

s=Storage(valDate,storageStart,storageEnd,curve=curve,n_p=0,v_step=v_step,sVol=vol)
n=len(s.date_span); active=np.ones(n); active[s._active:]=0.; active[:s.Dt]=0.
s.i_curve=active.copy(); s.w_curve=np.zeros(n)
full_cap=s.n_op_start; s.n_op_start=0
s.t_p_curve=np.full(s.n_op+2,-1e9); s.t_p_curve[full_cap]=0.
s.build()
s.set_volume_states(days); s.n_op_start=0
s.t_p_curve=np.full(s.n_op+2,-1e9); s.t_p_curve[days]=0.
s.build()
s.n_p=n_p_full; s.build()

Dt=s.Dt; n_op=s.n_op; n_p=n_p_full

print(f'n_op={n_op}, n_op_start={s.n_op_start}, days={days}')
print(f'Dt={Dt}, _active={s._active}')

# Print FULL k distribution at Jan 29 (day 28), Jan 30 (day 29), Jan 31 (day 30)
for day_offset, label in [(28, 'Jan 29 (Dt+28)'), (29, 'Jan 30 (Dt+29)'), (30, 'Jan 31 (Dt+30)')]:
    i = Dt + day_offset
    prob_i = s.prob[i]  # shape (61, n_op)
    print(f'\n=== {label}: i={i}, date={s.date_span[i].date()} ===')
    print(f'Full k distribution:')
    total = 0
    for k in range(n_op):
        prob_k = prob_i[:, k].sum()
        total += prob_k
        if prob_k > 1e-8:
            n_buy = (s.strat[i, :, k] == 1).sum()
            print(f'  k={k:2d}: prob={prob_k:.8f}  n_buy_states={n_buy}')
    print(f'  Total = {total:.8f}')

# Check strategy at k=29 on Jan 30 (day 29)
i_jan30 = Dt + 29
print(f'\n=== Strategy at k=29 on Jan 30 (i={i_jan30}) ===')
strat_k29 = s.strat[i_jan30, :, 29]
n_buy_k29 = (strat_k29 == 1).sum()
print(f'n_buy_states at k=29: {n_buy_k29} out of {2*n_p+1}')

# Check strategy at various k on Jan 31
i_jan31 = Dt + 30
print(f'\n=== Strategy summary at Jan 31 (i={i_jan31}) ===')
for k in range(n_op):
    n_buy = (s.strat[i_jan31, :, k] == 1).sum()
    prob_k = s.prob[i_jan31, :, k].sum()
    if prob_k > 1e-8 or k == 29:
        print(f'  k={k:2d}: n_buy={n_buy}, prob={prob_k:.8f}')

# Key: what is prob at k=29 and what exp_ex does it contribute?
prob_k29_jan30 = s.prob[i_jan30, :, 29].sum()
print(f'\nP(k=29 at Jan 30) = {prob_k29_jan30:.8f}')
print(f'P(k=30 at Jan 31) = {s.prob[i_jan31, :, n_op-1].sum():.8f}')

# Trace where k=30 prob on Jan 31 came from
# It should equal sum_j prob[Jan30, j, 29] * P(exercise at k=29) + prob[Jan30, j, 30] * 1
print('\n=== Tracing k=30 on Jan 31 ===')
acc_from_k29 = 0
acc_from_k30 = 0
for j in range(2*n_p+1):
    p_j29 = s.prob[i_jan30, j, 29]
    p_j30 = s.prob[i_jan30, j, 30]
    if p_j29 > 0:
        # k=29 can go to k=30 if strat=1
        if s.strat[i_jan30, j, 29] == 1:
            acc_from_k29 += p_j29
    if p_j30 > 0:
        acc_from_k30 += p_j30

# This only accounts for direct transition, need to account for price transitions too
print(f'Direct sum of prob[Jan30, j, 29] * P(exercise) = {acc_from_k29:.8f}')
print(f'Sum of prob[Jan30, j, 30] = {acc_from_k30:.8f}')
print(f'Actual P(k=30 at Jan 31) = {s.prob[i_jan31, :, n_op-1].sum():.8f}')
