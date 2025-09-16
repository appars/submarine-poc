# pages/2_Consume_and_Benchmark.py
import io, re, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Optional VTK reader (no GUI deps)
try:
    import meshio
    HAS_MESHIO = True
except Exception:
    HAS_MESHIO = False

st.set_page_config(page_title="Consume & Benchmark", layout="wide")
st.title("Consume & Benchmark: OpenFOAM results")
st.caption("Upload baseline & warm-start artifacts (log.simpleFoam, logs/, VTK/) to compute speedup and visualize fields.")

# ---------------- Parsing helpers ----------------
TIME_RE      = re.compile(r"^\s*Time\s*=\s*([0-9.eE+-]+)", re.M)  # OUTER iter tag
EXEC_TIME_RE = re.compile(r"ExecutionTime\s*=\s*([0-9.+-eE]+)\s*s")
SOLVE_RE     = re.compile(
    r"Solving for\s+([A-Za-z0-9_]+),\s*Initial residual\s*=\s*([0-9.eE+-]+),\s*Final residual\s*=\s*([0-9.eE+-]+).*?(?:No Iterations\s*=\s*(\d+))?"
)

def parse_log_simplefoam(text: str):
    """Return exec_time and DataFrame with columns: step(outer), field, init, final, nIters."""
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

def convergence_iter(df: pd.DataFrame, thresh: float):
    """
    First OUTER step where ALL standard fields present (Ux,Uy,Uz,p,k,omega) are <= thresh.
    If none of those exist, use all available fields.
    """
    if df.empty:
        return None
    preferred = ["Ux","Uy","Uz","p","k","omega"]
    avail = list(df["field"].unique())
    fields = [f for f in preferred if f in avail] or avail
    g = df[df["field"].isin(fields)].groupby("step")["final"].max().sort_index()
    ok = np.where(g.values <= thresh)[0]
    return int(g.index[ok[0]]) if len(ok) else None

# --------------- ZIP ingestion ---------------
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
    Returns: {"exec_time": float|None, "residuals": DataFrame, "vtu": list[(name, meshio.Mesh)]}
    """
    zf = zipfile.ZipFile(upload)

    # 1) Parse log.simpleFoam
    exec_time = None
    residuals = pd.DataFrame()
    for name in zf.namelist():
        if name.endswith("log.simpleFoam"):
            with zf.open(name) as f:
                text = f.read().decode("utf-8", errors="ignore")
            exec_time, residuals = parse_log_simplefoam(text)
            break

    # 2) Read VTU (latest time)
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

# --------------- Plotting ---------------
def plot_residuals(df: pd.DataFrame, title: str, thresh: float):
    """Log-scale curves for standard fields if present; otherwise plot all fields."""
    if df.empty:
        st.warning(f"{title}: no residuals parsed.")
        return

    preferred = ["Ux","Uy","Uz","p","k","omega"]
    avail = list(df["field"].unique())
    plot_fields = [f for f in preferred if f in avail] or sorted(avail)

    piv = df[df["field"].isin(plot_fields)].pivot_table(
        index="step", columns="field", values="final", aggfunc="last"
    ).sort_index()
    long = piv.reset_index().melt("step", var_name="field", value_name="final").dropna()
    if long.empty:
        st.info(f"{title}: no samples to plot.")
        return

    import plotly.express as px
    fig = px.line(long, x="step", y="final", color="field",
                  labels={"step":"Outer iteration (Time index)", "final":"Final residual"},
                  render_mode="webgl")
    fig.update_yaxes(type="log", tickformat=".1e")
    fig.add_hline(y=thresh, line_dash="dot", line_width=1)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

def field_slice_plot(meshes, field_name="U", component="mag", plane="Y", value=0.0, sample=6000):
    """Simple 3D scatter slice through the latest VTU set."""
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
            V = np.asarray(m.cell_data_dict[chosen][field_name])
            P = m.points[cells].mean(axis=1)

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
            if component == "mag": s = np.linalg.norm(V2, axis=1)
            elif component == "x": s = V2[:,0]
            elif component == "y": s = V2[:,1]
            else: s = V2[:,2]
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

# ---------------- UI ----------------
st.subheader("1) Upload artifacts")
c1, c2 = st.columns(2)
with c1:
    base_zip = st.file_uploader("Baseline ZIP (must contain log.simpleFoam; optional logs/ & VTK/)", type=["zip"], key="base")
with c2:
    warm_zip = st.file_uploader("Warm-start ZIP (must contain log.simpleFoam; optional logs/ & VTK/)", type=["zip"], key="warm")

thresh = st.number_input("Residual threshold (dashed line on plots)", min_value=1e-8, max_value=1e-3,
                         value=1e-5, step=1e-6, format="%.0e")

go = st.button("Compute metrics & visualize")
if not go:
    st.stop()

# Parse uploads
base = load_zip_artifacts(base_zip) if base_zip else None
warm = load_zip_artifacts(warm_zip) if warm_zip else None
if not base or base["residuals"].empty:
    st.error("Baseline residuals not found. Ensure your ZIP contains a valid log.simpleFoam."); st.stop()
if not warm or warm["residuals"].empty:
    st.error("Warm-start residuals not found. Ensure your ZIP contains a valid log.simpleFoam."); st.stop()

# ---------------- Residuals & metrics ----------------
st.divider()
st.subheader("2) Convergence & residuals (log scale)")

plot_residuals(base["residuals"], "Baseline", thresh)
b_iter = convergence_iter(base["residuals"], thresh)

plot_residuals(warm["residuals"], "Warm-start", thresh)
w_iter = convergence_iter(warm["residuals"], thresh)

b_time, w_time = base["exec_time"], warm["exec_time"]
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Baseline time (s)", f"{b_time:.3f}" if b_time else "n/a")
with m2: st.metric("Warm-start time (s)", f"{w_time:.3f}" if w_time else "n/a")
with m3: st.metric("Baseline iters (outer)", b_iter if b_iter is not None else "n/a")
with m4: st.metric("Warm-start iters (outer)", w_iter if w_iter is not None else "n/a")

if b_time and w_time:
    st.success(f"**Speedup** = baseline / warm-start = **{b_time / max(w_time, 1e-9):.2f}×**")

st.caption("Outer iteration index is parsed from 'Time = ...' in the log. The dashed line is your chosen residual threshold for reference.")

# ---------------- Force/Cd comparison ----------------
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
            if lc in ("cd","c_d") or "cd" in lc: return c
        return df.columns[-1]
    cb, cw = cd_col(dfb), cd_col(dfw)
    st.line_chart(pd.DataFrame({"baseline_Cd": dfb[cb], "warm_Cd": dfw[cw]}), height=260)
    if len(dfb[cb])>0 and len(dfw[cw])>0:
        st.write(f"Baseline mean Cd: **{dfb[cb].mean():.6f}**, Warm-start mean Cd: **{dfw[cw].mean():.6f}**")

# ---------------- VTK visualization ----------------
st.divider()
st.subheader("4) Field visualization from VTK (latest time)")

if not HAS_MESHIO:
    st.info("VTK viewing requires meshio; add `meshio` and `plotly` to requirements.txt.")
else:
    field = st.selectbox("Field", ["U","p","k","omega","nut"], index=0)
    comp  = st.selectbox("Component (vectors only)", ["mag","x","y","z"], index=0)
    plane = st.selectbox("Slice plane", ["Y","X","Z"], index=0)
    val   = st.number_input(f"{plane}-plane value (approx)", value=0.0)
    which = st.radio("Dataset", ["Warm-start","Baseline"], index=0, horizontal=True)

    meshes = warm["vtu"] if which == "Warm-start" else base["vtu"]
    if not meshes:
        st.warning("No VTU found in the ZIP (did you run `foamToVTK -latestTime` before zipping?).")
    else:
        field_slice_plot(meshes, field_name=field, component=comp, plane=plane, value=val, sample=6000)

