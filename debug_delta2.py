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
s.build(); flat_cost=s.flat()
s.set_volume_states(days); s.n_op_start=0
s.t_p_curve=np.full(s.n_op+2,-1e9); s.t_p_curve[days]=0.
s.build()
profiled_eur=s.v[0,0,0]; acq=-np.sum(s.delta)
s.n_p=n_p_full; s.build()

Dt=s.Dt; n_t=s.n_t; n_p=n_p_full

# Print detailed daily for first 20 active days
print('=== First 20 active days: date, fwd price, exp_ex, delta ===')
for i in range(Dt, Dt+20):
    d = s.date_span[i]
    fwd_i = s.fwd[i]
    ex_i = s.exp_ex[i]
    delta_i = s.delta[i]
    print(f'{d.date()}  fwd={fwd_i:.4f}  exp_ex={ex_i:.3f}  delta={delta_i:.3f}  ratio={delta_i/ex_i:.3f}')

# Print forward curve for Jan 2027
print('\n=== Forward curve Jan 2027 ===')
for i in range(Dt, Dt+31):
    print(f'{s.date_span[i].date()} fwd={s.fwd[i]:.6f}')

# Check: what fraction of price states exercise on day 0 of active period?
print('\n=== Strategy at day Dt (Jan 1): how many price states exercise (k=0)? ===')
n_buy = (s.strat[Dt, :, 0] == 1).sum()
n_hold = (s.strat[Dt, :, 0] == 0).sum()
print(f'Buy: {n_buy}, Hold: {n_hold} (out of {2*n_p+1} price states)')

print('\n=== Strategy at day Dt+7 (Jan 8): how many price states exercise (k=0)? ===')
n_buy2 = (s.strat[Dt+7, :, 0] == 1).sum()
n_hold2 = (s.strat[Dt+7, :, 0] == 0).sum()
print(f'Buy: {n_buy2}, Hold: {n_hold2}')

# Show prob distribution at Dt
prob_at_Dt = s.prob[Dt, :, :]
print(f'\n=== Prob distribution at day Dt (Jan 1): total={prob_at_Dt.sum():.6f} ===')
print('Price state distribution (summed over inventory):')
for j in range(2*n_p+1):
    p = prob_at_Dt[j, :].sum()
    if p > 0.001:
        strat_k0 = s.strat[Dt, j, 0]
        x_j = s.x[Dt, j]
        fwd_j = s.fwd[Dt]
        print(f'  j={j:2d}  prob={p:.4f}  x={x_j:.4f}  S/F={np.exp(x_j)/fwd_j:.3f}  strat={strat_k0:.0f}')

# Check probability sum at various times
print('\n=== Probability sums (should all be ~1.0) ===')
for i in [0, Dt-1, Dt, Dt+7, Dt+30, s._active-1]:
    print(f'i={i} ({s.date_span[i].date()}): sum={s.prob[i].sum():.6f}')
