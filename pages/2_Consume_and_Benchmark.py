# pages/2_Consume_and_Benchmark.py
import io, re, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Optional VTK reader (no heavy GUI deps)
try:
    import meshio
    HAS_MESHIO = True
except Exception:
    HAS_MESHIO = False

st.set_page_config(page_title="Consume & Benchmark", layout="wide")
st.title("Consume & Benchmark: OpenFOAM results")
st.caption(
    "Upload baseline & warm-start artifacts (log.simpleFoam, logs/, VTK/) to compute speedup, "
    "convergence iterations, and visualize fields."
)

# ---------- Log parsing ----------
TIME_RE        = re.compile(r"^\s*Time\s*=\s*([0-9.eE+-]+)", re.M)  # OUTER iteration tag
EXEC_TIME_RE   = re.compile(r"ExecutionTime\s*=\s*([0-9.+-eE]+)\s*s")
SOLVE_RE       = re.compile(
    r"Solving for\s+([A-Za-z0-9_]+),\s*Initial residual\s*=\s*([0-9.eE+-]+),\s*Final residual\s*=\s*([0-9.eE+-]+).*?(?:No Iterations\s*=\s*(\d+))?"
)

def parse_log_simplefoam(text: str):
    """Return exec_time and a DataFrame with columns: step(outer), field, init, final, nIters."""
    exec_times = [float(m.group(1)) for m in EXEC_TIME_RE.finditer(text)]
    exec_time = exec_times[-1] if exec_times else None

    step = -1
    rows = []
    for line in text.splitlines():
        if TIME_RE.match(line):
            step += 1
            continue
        m = SOLVE_RE.search(line)
        if m and step >= 0:
            fld   = m.group(1)
            initr = float(m.group(2))
            finr  = float(m.group(3))
            nits  = int(m.group(4)) if m.group(4) else math.nan
            rows.append((step, fld, initr, finr, nits))
    df = pd.DataFrame(rows, columns=["step", "field", "init", "final", "nIters"])
    return exec_time, df

def convergence_iter(df: pd.DataFrame, fields, thresh: float):
    """First OUTER step where ALL selected fields have final residual <= thresh."""
    if df.empty or not fields:
        return None
    df_ = df[df["field"].isin(fields)].copy()
    if df_.empty:
        return None
    g = df_.groupby("step")["final"].max().sort_index()
    ok = np.where(g.values <= thresh)[0]
    if len(ok) == 0:
        return None
    return int(g.index[ok[0]])

def first_cross_table(df: pd.DataFrame, fields, thresh: float) -> pd.DataFrame:
    """
    Per-field summary:
      Field | First ≤ thr (outer iter) | Best residual | Gap vs thr (× over/under)
    """
    rows = []
    for f in fields:
        sub = df[df["field"] == f]
        if sub.empty:
            rows.append((f, None, np.nan, np.nan))
            continue
        best = float(sub["final"].min())
        meets = sub.loc[sub["final"] <= thresh, "step"]
        first = int(meets.iloc[0]) if not meets.empty else None
        gap = (best / thresh) if (thresh > 0 and np.isfinite(best)) else np.nan  # >1 → above; <1 → under
        rows.append((f, first, best, gap))
    out = pd.DataFrame(rows, columns=["field", "first_iter", "best_final", "gap_vs_thresh"])
    # Nice formatting columns for display
    def fmt_iter(x): return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{int(x)}"
    def fmt_best(x): 
        try: return f"{x:.2e}"
        except: return "—"
    def fmt_gap(g):
        if g is None or (isinstance(g, float) and not np.isfinite(g)): return "—"
        if g <= 1: return f"{1/max(g,1e-12):.1f}× under"
        return f"{g:.1f}× over"
    out_disp = out.copy()
    out_disp["First ≤ thr (outer)"] = out_disp["first_iter"].map(fmt_iter)
    out_disp["Best residual"]       = out_disp["best_final"].map(fmt_best)
    out_disp["Gap vs thr"]          = out_disp["gap_vs_thresh"].map(fmt_gap)
    out_disp = out_disp[["field", "First ≤ thr (outer)", "Best residual", "Gap vs thr"]]
    return out_disp

# ---------- ZIP ingestion ----------
def _latest_time_from_paths(paths):
    def extract_num(p):
        try:
            nums = [float(t) for t in Path(p).parts if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", t)]
            return nums[-1] if nums else -1.0
        except Exception:
            return -1.0
    paths = list(paths)
    if not paths: return -1.0
    paths.sort(key=extract_num)
    return extract_num(paths[-1])

def load_zip_artifacts(upload) -> dict:
    """
    Returns dict: {"exec_time": float|None, "residuals": DataFrame, "vtu": list[(name, meshio.Mesh)]}
    """
    zf = zipfile.ZipFile(upload)

    exec_time = None
    residuals = pd.DataFrame()
    for name in zf.namelist():
        if name.endswith("log.simpleFoam"):
            with zf.open(name) as f:
                text = f.read().decode("utf-8", errors="ignore")
            exec_time, residuals = parse_log_simplefoam(text)
            break

    meshes = []
    if HAS_MESHIO:
        vtu_members = [n for n in zf.namelist() if n.lower().endswith(".vtu")]
        if vtu_members:
            latest = _latest_time_from_paths(vtu_members)
            latest_files = [n for n in vtu_members if f"{latest}" in n] or vtu_members[-5:]
            for n in latest_files:
                try:
                    with zf.open(n) as f:
                        meshes.append((n, meshio.read(io.BytesIO(f.read()))))
                except Exception:
                    pass
    return {"exec_time": exec_time, "residuals": residuals, "vtu": meshes}

# ---------- Plotting ----------
def plot_residuals(df: pd.DataFrame, title: str, plot_fields, thresh: float):
    if df.empty:
        st.warning(f"{title}: no residuals parsed.")
        return
    df_ = df[df["field"].isin(plot_fields)].copy()
    if df_.empty:
        st.warning(f"{title}: none of the selected fields were found in the log.")
        return
    piv = df_.pivot_table(index="step", columns="field", values="final", aggfunc="last").sort_index()
    long = piv.reset_index().melt("step", var_name="field", value_name="final").dropna()
    import plotly.express as px
    fig = px.line(long, x="step", y="final", color="field",
                  labels={"step":"Outer iteration (Time index)", "final":"Final residual"},
                  render_mode="webgl")
    fig.update_yaxes(type="log", tickformat=".1e")
    fig.add_hline(y=thresh, line_dash="dot", line_width=1)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

def field_slice_plot(meshes, field_name="U", component="mag", plane="Y", value=0.0, sample=6000):
    import plotly.express as px
    rng = np.random.default_rng(0)
    all_pts, all_vals = [], []
    for _, m in meshes:
        if field_name in m.point_data:
            V = np.asarray(m.point_data[field_name]); P = m.points
        else:
            if not m.cell_data_dict: continue
            chosen = next((ct for ct in m.cell_data_dict if field_name in m.cell_data_dict[ct]), None)
            if chosen is None: continue
            cells = next((cb.data for cb in m.cells if cb.type == chosen), None)
            if cells is None: continue
            V = np.asarray(m.cell_data_dict[chosen][field_name]); P = m.points[cells].mean(axis=1)
        x, y, z = P[:,0], P[:,1], P[:,2]
        if plane == "Y":
            tol = 0.01*(np.nanmax(np.abs(y)) or 1.0) + 1e-6; mask = np.isclose(y, value, atol=tol)
        elif plane == "X":
            tol = 0.01*(np.nanmax(np.abs(x)) or 1.0) + 1e-6; mask = np.isclose(x, value, atol=tol)
        else:
            tol = 0.01*(np.nanmax(np.abs(z)) or 1.0) + 1e-6; mask = np.isclose(z, value, atol=tol)
        P2, V2 = P[mask], V[mask]
        if P2.size == 0: continue
        if V2.ndim == 2 and V2.shape[1] == 3:
            s = np.linalg.norm(V2, axis=1) if component == "mag" else V2[:, {"x":0,"y":1,"z":2}[component]]
        else:
            s = V2
        n = len(P2)
        if n > sample:
            idx = rng.choice(n, size=sample, replace=False)
            P2, s = P2[idx], s[idx]
        all_pts.append(P2); all_vals.append(s)
    if not all_pts:
        st.warning("Could not find the selected field in uploaded VTK files (or slice plane has no points).")
        return
    P = np.vstack(all_pts); S = np.concatenate(all_vals)
    df = pd.DataFrame({"x": P[:,0], "y": P[:,1], "z": P[:,2], "val": S})
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="val", opacity=0.65, height=520)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- UI -----------------------------
st.subheader("1) Upload artifacts")
c1, c2 = st.columns(2)
with c1:
    base_zip = st.file_uploader("Baseline ZIP (must contain log.simpleFoam; optional logs/ & VTK/)", type=["zip"], key="base")
with c2:
    warm_zip = st.file_uploader("Warm-start ZIP (must contain log.simpleFoam; optional logs/ & VTK/)", type=["zip"], key="warm")

thresh = st.number_input("Residual threshold for convergence", min_value=1e-8, max_value=1e-3, value=1e-5, step=1e-6, format="%.0e")
go = st.button("Compute metrics & visualize")
if not go:
    st.stop()

# ----------------------------- parse & metrics -----------------------------
base = load_zip_artifacts(base_zip) if base_zip else None
warm = load_zip_artifacts(warm_zip) if warm_zip else None

if not base or base["residuals"].empty:
    st.error("Baseline residuals not found. Ensure your ZIP contains a valid log.simpleFoam.")
    st.stop()
if not warm or warm["residuals"].empty:
    st.error("Warm-start residuals not found. Ensure your ZIP contains a valid log.simpleFoam.")
    st.stop()

# Field selection for convergence rule
st.divider()
st.subheader("2) Convergence & residuals")
all_fields = sorted(set(base["residuals"]["field"].unique()) | set(warm["residuals"]["field"].unique()))
default_fields = [f for f in ["Ux","Uy","Uz","p","k","omega"] if f in all_fields]
chosen = st.multiselect("Fields that must be ≤ threshold to declare a run 'converged'", options=all_fields, default=default_fields)

# Plots
plot_fields = [f for f in ["Ux","Uy","Uz","p","k","omega"] if f in all_fields] or all_fields
plot_residuals(base["residuals"], "Baseline", plot_fields, thresh)
b_iter = convergence_iter(base["residuals"], chosen, thresh)

if b_iter is not None:
    st.success(f"Baseline: convergence (selected fields ≤ {thresh:g}) by outer iter **{b_iter}**.")
else:
    st.info(f"Baseline: did not meet selected threshold {thresh:g} within parsed iterations.")

plot_residuals(warm["residuals"], "Warm-start", plot_fields, thresh)
w_iter = convergence_iter(warm["residuals"], chosen, thresh)
if w_iter is not None:
    st.success(f"Warm-start: convergence (selected fields ≤ {thresh:g}) by outer iter **{w_iter}**.")
else:
    st.info(f"Warm-start: did not meet selected threshold {thresh:g} within parsed iterations.")

# Per-field tables (first crossing + gap to threshold)
st.markdown("##### Per-field first crossing & gap")
c3, c4 = st.columns(2)
with c3:
    st.caption("Baseline")
    st.dataframe(first_cross_table(base["residuals"], chosen or all_fields, thresh), use_container_width=True)
with c4:
    st.caption("Warm-start")
    st.dataframe(first_cross_table(warm["residuals"], chosen or all_fields, thresh), use_container_width=True)

# High-level metrics
b_time, w_time = base["exec_time"], warm["exec_time"]
c5, c6, c7, c8 = st.columns(4)
with c5: st.metric("Baseline time (s)", f"{b_time:.3f}" if b_time else "n/a")
with c6: st.metric("Warm-start time (s)", f"{w_time:.3f}" if w_time else "n/a")
with c7: st.metric("Baseline iters (outer)", b_iter if b_iter is not None else "n/a")
with c8: st.metric("Warm-start iters (outer)", w_iter if w_iter is not None else "n/a")

if b_time and w_time:
    st.success(f"**Speedup** = baseline / warm-start = **{b_time / max(w_time, 1e-9):.2f}×**")

st.caption("Notes: (1) Iteration index = SIMPLE **outer** step parsed from 'Time = ...'. "
           "(2) ExecutionTime is parsed from the final line of log.simpleFoam. "
           "(3) Threshold line helps visualize the chosen convergence target.")

# ----------------------------- Cd comparison (optional) -----------------------------
st.divider()
st.subheader("3) Force/Cd comparison (optional)")
d1, d2 = st.columns(2)
with d1:
    base_cd = st.file_uploader("Baseline forces/Cd CSV", type=["csv"], key="bcd")
with d2:
    warm_cd = st.file_uploader("Warm-start forces/Cd CSV", type=["csv"], key="wcd")
if base_cd and warm_cd:
    dfb = pd.read_csv(base_cd); dfw = pd.read_csv(warm_cd)
    def cd_col(df):
        for c in df.columns:
            lc = c.lower()
            if lc in ("cd", "c_d") or "cd" in lc: return c
        return df.columns[-1]
    cb, cw = cd_col(dfb), cd_col(dfw)
    st.line_chart(pd.DataFrame({"baseline_Cd": dfb[cb], "warm_Cd": dfw[cw]}), height=260)
    if len(dfb[cb])>0 and len(dfw[cw])>0:
        st.write(f"Baseline mean Cd: **{dfb[cb].mean():.6f}**, Warm-start mean Cd: **{dfw[cw].mean():.6f}**")

# ----------------------------- VTK visualization -----------------------------
st.divider()
st.subheader("4) Field visualization from VTK (latest time)")
if not HAS_MESHIO:
    st.info("VTK viewing requires meshio; add `meshio` and `plotly` to requirements.txt.")
else:
    field = st.selectbox("Field", ["U", "p", "k", "omega", "nut"], index=0)
    comp  = st.selectbox("Component (vectors only)", ["mag", "x", "y", "z"], index=0)
    plane = st.selectbox("Slice plane", ["Y", "X", "Z"], index=0)
    val   = st.number_input(f"{plane}-plane value (approx)", value=0.0)
    which = st.radio("Dataset", ["Warm-start", "Baseline"], index=0, horizontal=True)
    meshes = warm["vtu"] if which == "Warm-start" else base["vtu"]
    if not meshes:
        st.warning("No VTU found in the ZIP (did you run `foamToVTK -latestTime` before zipping?).")
    else:
        field_slice_plot(meshes, field_name=field, component=comp, plane=plane, value=val, sample=6000)

st.caption("Tip: Select which fields define 'converged' above. Tables show the first crossing and the residual gap vs threshold.")

