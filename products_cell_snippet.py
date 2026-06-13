# Optional: drive forward.ipynb from products.xlsx instead of its hard-coded
# "Inputs" cell. Paste this in place of the Inputs cell — it sets the same
# product_type / dates / vol / grid variables, loaded from the named product row,
# so the valuation cell below it runs unchanged.
#
# To call the model directly from your own code, prefer the cleaner library helper
# (no notebook needed):
#     from storage_model import load_product_params, params_for_run_valuation, run_valuation
#     params = params_for_run_valuation(load_product_params("products.xlsx", PRODUCT))
#     s, result = run_valuation(curve, params)

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
