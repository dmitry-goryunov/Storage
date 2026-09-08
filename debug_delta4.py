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

# For Jan 30 (Dt+29=394) and Jan 31 (Dt+30=395), show strategy for k=0..10
# and probability distribution
for day_offset, label in [(29, 'Jan 30 (Dt+29)'), (30, 'Jan 31 (Dt+30)'), (31, 'Feb 1 (Dt+31)')]:
    i = Dt + day_offset
    prob_i = s.prob[i]  # shape (61, n_op)
    print(f'\n=== {label}: i={i}, date={s.date_span[i].date()} ===')
    print(f'Probability at each inventory state k (summed over j):')
    for k in range(min(n_op, 10)):
        prob_k = prob_i[:, k].sum()
        n_buy = (s.strat[i, :, k] == 1).sum()
        if prob_k > 1e-6:
            print(f'  k={k:2d}: prob={prob_k:.6f}  strat_buy_states={n_buy}')
    prob_total = prob_i.sum()
    print(f'  Total prob = {prob_total:.6f}')

    # Compute exp_ex for this day manually
    action_k = np.zeros(n_op)
    for k in range(n_op):
        inj_step = 1 if k < days else 0
        for j in range(2*n_p+1):
            if s.strat[i, j, k] == 1:
                action_k[k] = inj_step * v_step
                break

    # Manual exp_ex computation
    manual_exp_ex = 0.0
    for j in range(2*n_p+1):
        for k in range(n_op):
            inj_step = 1 if k < days else 0
            if s.strat[i, j, k] == 1:
                manual_exp_ex -= inj_step * v_step * prob_i[j, k]
    print(f'Manual exp_ex = {manual_exp_ex:.3f} (model: {s.exp_ex[i]:.3f})')

print('\n=== Summary: daily probability at k=0 vs total prob and exp_ex ===')
print(f"{'day':>5} {'date':<12} {'prob_k0':>10} {'prob_k30':>10} {'exp_ex':>10}")
for off in [0,1,2,3,4,5,6,7,14,21,29,30,31,35,38]:
    i = Dt + off
    p_k0 = s.prob[i, :, 0].sum()
    p_k30 = s.prob[i, :, -1].sum() if n_op > 1 else 0
    print(f'{off:>5} {str(s.date_span[i].date()):<12} {p_k0:>10.6f} {p_k30:>10.8f} {s.exp_ex[i]:>10.3f}')
