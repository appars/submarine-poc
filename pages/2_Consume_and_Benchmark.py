import io, os, re, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Optional lightweight VTK reader for unstructured .vtu
# (meshio avoids heavy VTK GUI deps and works on Streamlit Cloud)
try:
    import meshio
    HAS_MESHIO = True
except Exception:
    HAS_MESHIO = False

st.set_page_config(page_title="Consume & Benchmark", layout="wide")

st.title("Consume & Benchmark: OpenFOAM results")
st.caption("Upload baseline & warm-start artifacts (log.simpleFoam, logs/, VTK/) to compute speedup & visualize fields.")

# ----------------------------- helpers -----------------------------
EXEC_TIME_RE = re.compile(r"ExecutionTime\s*=\s*([0-9.+-eE]+)\s*s")
SOLVE_RE = re.compile(
    r"Solving for\s+([A-Za-z0-9_]+),\s*Initial residual\s*=\s*([0-9.eE+-]+),\s*Final residual\s*=\s*([0-9.eE+-]+).*?(?:No Iterations\s*=\s*(\d+))?"
)

def parse_log_simplefoam(text: str):
    """Return dict with execution time (s) and a residual timeline dataframe."""
    exec_times = [float(m.group(1)) for m in EXEC_TIME_RE.finditer(text)]
    exec_time = exec_times[-1] if exec_times else None

    # build residual table in 'iteration' order
    iters, field, init_res, final_res, itcount = [], [], [], [], []
    i = 0
    for m in SOLVE_RE.finditer(text):
        i += 1
        fld = m.group(1)
        init_r = float(m.group(2))
        fin_r  = float(m.group(3))
        count  = int(m.group(4)) if m.group(4) else np.nan
        iters.append(i); field.append(fld); init_res.append(init_r); final_res.append(fin_r); itcount.append(count)
    df = pd.DataFrame({"iter": iters, "field": field, "init": init_res, "final": final_res, "nIters": itcount})
    return exec_time, df

def convergence_iter(df: pd.DataFrame, fields=("Ux","Uy","Uz","p","k","omega"), thresh=1e-5):
    """First iteration index where *all* specified fields have final residual <= thresh."""
    # group by iteration and check max residual among chosen fields
    df_ = df[df["field"].isin(fields)]
    if df_.empty:
        return None
    # at each 'iter' compute max final residual across fields
    g = df_.groupby("iter")["final"].max()
    # find first iter where max <= thresh and stays below afterwards
    idx = np.where(g.values <= thresh)[0]
    if len(idx) == 0:
        return None
    return int(g.index[idx[0]])

def load_zip_artifacts(upload) -> dict:
    """
    Accepts a zip upload; returns:
      - exec_time (float or None)
      - residual_df (DataFrame)
      - vtu_files (list of (name, meshio.Mesh) if meshio available)
    """
    zf = zipfile.ZipFile(upload)
    # 1) log.simpleFoam
    exec_time = None; res_df = pd.DataFrame()
    for name in zf.namelist():
        if name.endswith("log.simpleFoam"):
            with zf.open(name) as f:
                txt = f.read().decode("utf-8", errors="ignore")
            exec_time, res_df = parse_log_simplefoam(txt)
            break

    # 2) VTK .vtu files (latest time)
    vtu_members = [n for n in zf.namelist() if n.lower().endswith(".vtu")]
    meshes = []
    if HAS_MESHIO and vtu_members:
        # pick the time folder with max numeric name if present
        def time_of(path):
            parts = Path(path).parts
            try:
                # find a numeric part like '1000' or '50.0'
                nums = [float(p) for p in parts if re.match(r"^[0-9.]+$", p)]
                return nums[-1] if nums else -1.0
            except:
                return -1.0
        vtu_members.sort(key=time_of)
        latest_time = time_of(vtu_members[-1])
        latest = [n for n in vtu_members if str(latest_time) in n]
        for name in latest:
            with zf.open(name) as f:
                data = io.BytesIO(f.read())
            try:
                meshes.append((name, meshio.read(data)))
            except Exception:
                pass
    return {"exec_time": exec_time, "residuals": res_df, "vtu": meshes}

def summarize_residuals(df: pd.DataFrame, title: str, thresh: float):
    if df.empty:
        st.warning(f"{title}: no residuals parsed.")
        return None
    # pivot for plotting: iter x field -> final
    piv = df.pivot_table(index="iter", columns="field", values="final", aggfunc="last")
    st.line_chart(piv, height=280)
    conv_at = convergence_iter(df, thresh=thresh)
    if conv_at:
        st.success(f"{title}: convergence (all fields ≤ {thresh:g}) by iter **{conv_at}**.")
    else:
        st.info(f"{title}: did not meet threshold {thresh:g} within parsed iterations.")
    return conv_at

def field_slice_plot(meshes, field_name="U", component="mag", plane="Y", value=0.0, sample=5000):
    """Scatter slice colored by magnitude/selected component."""
    import plotly.express as px
    pts = []; cols = []
    for _, m in meshes:
        if field_name not in m.point_data and field_name not in m.cell_data_dict.get("Triangle", {}):
            continue
        # Prefer point_data
        if field_name in m.point_data:
            V = np.array(m.point_data[field_name])
            P = m.points
        else:
            # fallback: take cell centers for the first cell block with field
            key = list(m.cell_data_dict["Triangle"].keys())[0]
            cells = m.cells[0].data
            V = np.array(m.cell_data_dict["Triangle"][field_name])
            P = m.points[cells].mean(axis=1)

        # filter by plane
        x, y, z = P[:,0], P[:,1], P[:,2]
        if plane == "Y":
            mask = np.isclose(y, value, atol=0.01*np.nanmax(np.abs(y)) + 1e-6)
        elif plane == "X":
            mask = np.isclose(x, value, atol=0.01*np.nanmax(np.abs(x)) + 1e-6)
        else:
            mask = np.isclose(z, value, atol=0.01*np.nanmax(np.abs(z)) + 1e-6)
        P2 = P[mask]
        V2 = V[mask]
        if len(P2) == 0:
            continue

        # pick scalar
        if V2.ndim == 2 and V2.shape[1] == 3:
            if component == "mag":
                s = np.linalg.norm(V2, axis=1)
            elif component == "x": s = V2[:,0]
            elif component == "y": s = V2[:,1]
            else: s = V2[:,2]
        else:
            s = V2

        # downsample
        n = len(P2)
        if n > sample:
            idx = np.random.default_rng(0).choice(n, size=sample, replace=False)
            P2 = P2[idx]; s = s[idx]

        pts.append(P2); cols.append(s)

    if not pts:
        st.warning("Could not find the selected field in uploaded VTK files.")
        return
    P = np.vstack(pts); s = np.concatenate(cols)
    df = pd.DataFrame({"x": P[:,0], "y": P[:,1], "z": P[:,2], "val": s})
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="val", opacity=0.6, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- UI -----------------------------
st.subheader("1) Upload artifacts")

c1, c2 = st.columns(2)
with c1:
    base_zip = st.file_uploader("Baseline ZIP (log.simpleFoam + logs/ + VTK/)", type=["zip"], key="base")
with c2:
    warm_zip = st.file_uploader("Warm-start ZIP (log.simpleFoam + logs/ + VTK/)", type=["zip"], key="warm")

thresh = st.number_input("Residual threshold for convergence", 1e-8, 1e-3, value=1e-5, step=1e-6, format="%.0e")

go = st.button("Compute metrics & visualize")
if not go:
    st.stop()

# ----------------------------- parse & metrics -----------------------------
base = load_zip_artifacts(base_zip) if base_zip else None
warm = load_zip_artifacts(warm_zip) if warm_zip else None

if not base or base["residuals"].empty:
    st.error("Baseline residuals not found. Ensure log.simpleFoam is inside the ZIP.")
    st.stop()
if not warm or warm["residuals"].empty:
    st.error("Warm-start residuals not found. Ensure log.simpleFoam is inside the ZIP.")
    st.stop()

st.divider()
st.subheader("2) Convergence & residuals")

b_iter = summarize_residuals(base["residuals"], "Baseline", thresh)
w_iter = summarize_residuals(warm["residuals"], "Warm-start", thresh)

b_time = base["exec_time"]
w_time = warm["exec_time"]

c3, c4, c5, c6 = st.columns(4)
with c3: st.metric("Baseline time (s)", f"{b_time:.3f}" if b_time else "n/a")
with c4: st.metric("Warm-start time (s)", f"{w_time:.3f}" if w_time else "n/a")
with c5: st.metric("Baseline iters", b_iter if b_iter else "n/a")
with c6: st.metric("Warm-start iters", w_iter if w_iter else "n/a")

if b_time and w_time:
    speedup = b_time / max(w_time, 1e-9)
    st.success(f"**Speedup** = baseline / warm-start = **{speedup:.2f}×**")

st.caption("Note: ExecutionTime is parsed from log.simpleFoam. Iteration index is an approximate outer-iteration counter from solver lines.")

# ----------------------------- Cd comparison (optional) -----------------------------
st.divider()
st.subheader("3) Force/Cd comparison (optional)")

c7, c8 = st.columns(2)
with c7:
    base_cd = st.file_uploader("Baseline forces/Cd CSV (optional)", type=["csv"], key="bcd")
with c8:
    warm_cd = st.file_uploader("Warm-start forces/Cd CSV (optional)", type=["csv"], key="wcd")

if base_cd and warm_cd:
    dfb = pd.read_csv(base_cd)
    dfw = pd.read_csv(warm_cd)
    # Guess Cd column
    def cd_col(df):
        for c in df.columns:
            if c.lower() in ("cd","c_d") or "cd" in c.lower():
                return c
        return df.columns[-1]
    cb = cd_col(dfb); cw = cd_col(dfw)
    st.line_chart(pd.DataFrame({"baseline_Cd": dfb[cb], "warm_Cd": dfw[cw]}), height=260)
    if len(dfb[cb])>0 and len(dfw[cw])>0:
        st.write(f"Baseline mean Cd: **{dfb[cb].mean():.6f}**, Warm-start mean Cd: **{dfw[cw].mean():.6f}**")

# ----------------------------- VTK visualization -----------------------------
st.divider()
st.subheader("4) Field visualization from VTK (latest time)")

if not HAS_MESHIO:
    st.info("VTK viewing requires meshio; ask your admin to include `meshio` and `plotly` in requirements.txt")
else:
    field = st.selectbox("Field", ["U","p","k","omega","nut","TI"], index=0)
    comp  = "mag" if field=="U" else "mag"
    plane = st.selectbox("Slice plane", ["Y","X","Z"], index=0)
    val   = st.number_input(f"{plane}-plane value (approx)", value=0.0)
    which = st.radio("Dataset", ["Warm-start","Baseline"], index=0, horizontal=True)
    meshes = warm["vtu"] if which=="Warm-start" else base["vtu"]
    if not meshes:
        st.warning("No VTU found in the ZIP (did you run foamToVTK -latestTime?).")
    else:
        field_slice_plot(meshes, field_name=field, component=comp, plane=plane, value=val, sample=6000)

st.caption("Tip: To color by speed, select field = U (vector) with component=mag (default). For scalars (p, k, omega), the magnitude equals the value.")

