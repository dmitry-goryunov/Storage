# ── Paste this over the hard-coded parameter block at the top of the
# ── valuation cell in forward.ipynb (everything down to and including the
# ── "ratchets = ..." line). The Grid print and the rest of the cell stay.

PRODUCT = "call_swing_2010"        # row name in products.xlsx, sheet "products"

prm = storage_model.load_product_params("products.xlsx", PRODUCT)

product_type  = prm["product_type"]
FDDate        = prm["FDDate"]
valDate       = prm["valDate"]
storageStart  = prm["storageStart"]
storageEnd    = prm["storageEnd"]
vol           = prm["vol"]
n_p_full      = prm["n_p_full"]
run_intrinsic = prm["run_intrinsic"]

capacity_mwh         = prm["capacity_mwh"]
initial_storage_mwh  = prm["initial_storage_mwh"]
terminal_storage_mwh = prm["terminal_storage_mwh"]
inj_days      = prm["inj_days"]
wdr_days      = prm["wdr_days"]
n_states      = prm["n_states"]
inj_cost      = prm["inj_cost"]
wdr_cost      = prm["wdr_cost"]

ratchets      = prm["ratchets"]          # None, or (fullness, inj, wdr) from the named profile
use_ratchets  = ratchets is not None

print(f"Product: {PRODUCT} ({product_type}) | {prm['notes']}")

# ── derived grid (unchanged) ──
v_step   = capacity_mwh / n_states
initial_inv = int(round(initial_storage_mwh / v_step))
terminal_inv = int(round(terminal_storage_mwh / v_step))
if not 0 <= initial_inv <= n_states:
    raise ValueError(f"initial_storage_mwh must be between 0 and {capacity_mwh:,.0f} MWh")
if not 0 <= terminal_inv <= n_states:
    raise ValueError(f"terminal_storage_mwh must be between 0 and {capacity_mwh:,.0f} MWh")
inj_rate = max(1, int(round(n_states / inj_days)))   # base inject clips/day (pre-ratchet)
wdr_rate = max(1, int(round(n_states / wdr_days)))   # base withdraw clips/day (pre-ratchet)
