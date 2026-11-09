"""
Script to rebuild and execute feature-analysis.ipynb.
Run this after restart (or whenever the notebook appears empty) to regenerate it.
It:
  1. Builds the full notebook (imports, cleaning, plot helpers, 14 numeric, id, target,
     10 categorical, two heatmaps, numeric & categorical summary tables).
  2. Executes every cell via nbclient (Jupyter kernel).
  3. Saves feature-analysis.ipynb with all outputs (plots, tables).
"""
import os, sys, warnings, json, nbformat
warnings.filterwarnings("ignore")

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "feature-analysis.ipynb")
BACKUP_PATH = NOTEBOOK_PATH + ".bak"

# ── 1️⃣ Build notebook ────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

def md(s):
    nb["cells"].append(nbf.v4.new_markdown_cell(s.strip("\n")))

def code(s):
    nb["cells"].append(nbf.v4.new_code_cell(s.strip("\n")))

# ── Title + strategy
md("# Feature Analysis — Every Column vs the Target")
md("**Target:** `classification` → `ckd` (250, 62.5 %) / `notckd` (150, 37.5 %)")
md("## Plot strategy (target is categorical)")
md("| Condition | Plot | Why |\n|---|---|---|\n")
md("| Few discrete levels (≤ 6), ordinal | **Bar plot** (level counts by target) | The mode is the meaningful summary; box/violin smear an ordinal scale. |")
md("| Continuous, `mean ≈ median`, `|skew| < 0.5 | **Violin plot** | Near‑symmetric, so the full distribution shape can be compared by class. |")
md("| Continuous, `mean ≠ median`, `|skew| ≥ 0.5 | **Box plot** | Median/IQR is robust to skewed tails and outliers. |")
md("| Categorical vs categorical | **Grouped count plots** + **heatmaps** of level‑vs‑target split and Cramér’s V | One view per feature, plus an association matrix. |")

# ── 0️⃣ Setup + clean
md("## 0. Setup — load and clean")
code("""%matplotlib inline
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["axes.titlesize"] = 12

DATA_PATH = "kidney_disease.csv"
raw = pd.read_csv(DATA_PATH)
clean = raw.copy()
text_cols = [c for c in clean.columns if not pd.api.types.is_numeric_dtype(clean[c])]
for c in text_cols:
    clean[c] = (clean[c].astype("string")
                .str.replace("\\t", "", regex=False).str.strip())
    clean[c] = clean[c].replace({"": pd.NA, "?": pd.NA, "nan": pd.NA})
for c in ["pcv", "wc", "rc"]:
    clean[c] = pd.to_numeric(clean[c], errors="coerce").astype("float64")
clean["classification"] = clean["classification"].map({"ckd": 1, "notckd": 0}).astype("int64")
clean["target"] = clean["classification"].map({0: "notckd", 1: "ckd"})
TARGET = "target"
ORDER = ["notckd", "ckd"]
PALETTE = {"notckd": "#66C2A5", "ckd": "#FC8D62"}
NUM_COLS = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
CAT_COLS = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
PLOT_FOR = {"age":"box","bp":"box","bgr":"box","bu":"box","sc":"box","sod":"box","pot":"box","wc":"box",
            "hemo":"violin","pcv":"violin","rc":"violin","sg":"bar","al":"bar","su":"bar"}
print(f"{clean.shape[0]} rows x {raw.shape[1]} columns (raw); analysis frame: {clean.shape[1]} cols")
print("Target counts:")
print(clean[TARGET].value_counts().reindex(ORDER).to_string())
""")

# ── 1️⃣ Plot helpers
md("## 1. Plot helpers")
code("""def plot_box(col):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=clean, x=TARGET, y=col, order=ORDER, hue=TARGET, hue_order=ORDER,
                palette=PALETTE, width=0.5, dodge=False, legend=False, ax=ax)
    sns.stripplot(data=clean, x=TARGET, y=col, order=ORDER, color="black",
                  size=2, alpha=0.30, jitter=True, ax=ax)
    ax.set_title(f"{col} by target — box plot"); ax.set_xlabel(""); ax.set_ylabel(col)
    plt.tight_layout(); plt.show()

def plot_violin(col):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=clean, x=TARGET, y=col, order=ORDER, hue=TARGET, hue_order=ORDER,
                   palette=PALETTE, cut=0, inner="quartile", dodge=False, legend=False, ax=ax)
    ax.set_title(f"{col} by target — violin plot"); ax.set_xlabel(""); ax.set_ylabel(col)
    plt.tight_layout(); plt.show()

def plot_ordinal_bars(col):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=clean, x=col, hue=TARGET, hue_order=ORDER, palette=PALETTE, ax=ax)
    ax.set_title(f"{col} level counts by target — bar plot"); ax.set_xlabel(col); ax.set_ylabel("patients")
    plt.tight_layout(); plt.show()

def plot_count(col):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=clean, x=col, hue=TARGET, hue_order=ORDER, palette=PALETTE, ax=ax)
    ax.set_title(f"{col} by target — count plot"); ax.set_xlabel(col); ax.set_ylabel("patients")
    plt.tight_layout(); plt.show()

def cramers_v(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    ct = pd.crosstab(df["a"], df["b"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    observed = ct.to_numpy(); n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    r, k = observed.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))
""")

# ── 2️⃣ Numeric features vs target
md("## 2. Numeric features vs target")
numeric = [
    ("age","box","mean 51.5 < median 55 < mode 60 → left‑skewed; box keeps median/IQR."),
    ("bp","box","median 80 vs 70 mmHg; right‑tail outliers → box + dots."),
    ("sg","bar","only 5 discrete levels; mode 1.02; bar preserves ordinal levels."),
    ("al","bar","ordinal 0–5, zero‑inflated; per‑grade counts beat a single mean."),
    ("su","bar","median/mode 0 for both classes; mean 0.77 vs 0 only shows in bars."),
    ("bgr","box","mean 175 vs 108, skew +2.0 → box/median robust."),
    ("bu","box","median 53 vs 33, skew +2.6 → box isolates extremes."),
    ("sc","box","median 2.25 vs 0.90, skew +7.5 → median/IQR the honest comparison."),
    ("sod","box","median 136 vs 141; `4.5` typo appears as low outlier; box robust."),
    ("pot","box","median 4.3 vs 4.5, mean 4.88 vs 4.34; `47` typo appears as outlier; box robust."),
    ("hemo","violin","mean ≈ median, |skew| < 0.5 → violin shows level shift + shape."),
    ("pcv","violin","near‑symmetric (mean 38.9 ≈ median 40, skew −0.4) → violin compares distributions."),
    ("rc","violin","symmetric (mean 4.71 ≈ median 4.80, skew −0.2) → violin; note multicollinearity with hemo/pcv."),
    ("wc","box","mean 9070 vs 7706, skew +1.6 → box keeps median meaningful against tail."),
]
for col, kind, why in numeric:
    md(f"### `{col}` (numeric) vs `classification` — {kind} plot\n\n{why}")
    func = {"box": "plot_box", "violin": "plot_violin", "bar": "plot_ordinal_bars"}[kind]
    code(f'{func}("{col}")')

# ── id + target
md("### `id` (identifier) vs `classification` — box plot")
md("**What the plot says:** the two classes have identical distributions of `id`; it is just a sequential row number (0–399).\n\n**Why a box plot:** it is the quickest way to confirm the null — `id` carries no clinical signal and should be **dropped** before modeling.")
code("""fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=clean, x=TARGET, y="id", order=ORDER, hue=TARGET, hue_order=ORDER,
            palette=PALETTE, dodge=False, legend=False, ax=ax)
sns.stripplot(data=clean, x=TARGET, y="id", order=ORDER, color="black",
              size=2, alpha=0.25, jitter=True, ax=ax)
ax.set_title("id by target — box plot"); ax.set_xlabel(""); ax.set_ylabel("id")
plt.tight_layout(); plt.show()
""")

md("### `classification` (target) — class balance")
md("**What the plot says:** **250 CKD (62.5 %)** vs **150 not‑CKD (37.5 %)** — moderately imbalanced.\n\n**Why a count plot:** the target is categorical, so its own distribution is best shown as counts; the imbalance means we should use a stratified split and class weights / resampling.")
code("""fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=clean, x=TARGET, order=ORDER, hue=TARGET, hue_order=ORDER,
              palette=PALETTE, legend=False, ax=ax)
ax.bar_label(ax.containers[0])
ax.set_title("Target class balance"); ax.set_xlabel(""); ax.set_ylabel("patients")
plt.tight_layout(); plt.show()
print(clean[TARGET].value_counts(normalize=True).reindex(ORDER).mul(100).round(1).to_string())
""")

# ── 3️⃣ Categorical features vs target
md("## 3. Categorical features vs target")
cat_cells = [
    ("rbc","**What the plot says:** all **47 `abnormal` RBC cases are CKD (100 %)**, while `normal` is 70% not‑CKD. **38 % missing**.\n\n**Why a count plot:** binary categorical vs categorical → grouped counts expose the class split; the missingness is target‑related (MNAR), so keep an `rbc_was_missing` flag."),
    ("pc","**What the plot says:** all **76 `abnormal` pus‑cell cases are CKD (100 %)**; `normal` is 54% not‑CKD / 46 % CKD. **16 % missing**.\n\n**Why a count plot:** a binary variable against the target shows the clean split at a glance, and pairs with the missing‑value flag."),
    ("pcc","**What the plot says:** all **42 `present` pus‑cell‑clump cases are CKD (100 %)**; `notpresent` is 59 % CKD / 41 % not‑CKD.\n\n**Why a count plot:** low‑cardinality categorical vs target → grouped counts; the positive level is an almost perfect separator (validate it on a hold‑out split)."),
    ("ba","**What the plot says:** all **22 `present` bacteria cases are CKD (100 %)**; `notpresent` is 61 % CKD / 39 % not‑CKD.\n\n**Why a count plot:** a rare positive level that is CKD‑exclusive — the count plot makes both the rarity and the purity obvious."),
    ("htn","**What the plot says:** **every hypertensive patient is CKD (147/147, 100 %)**; among non‑hypertensives, 59 % are not‑CKD.\n\n**Why a count plot:** binary clinical flag vs binary target → grouped counts; hypertension is strongly CKD‑linked (watch for leakage / selection bias)."),
    ("dm","**What the plot says:** **every diabetic patient is CKD (137/137, 100 %)**; non‑diabetics are 57 % not‑CKD / 43 % CKD.\n\n**Why a count plot:** binary flag vs target; diabetes is a known CKD driver, so the association is clinically plausible but near‑separating."),
    ("cad","**What the plot says:** all **34 `yes` coronary‑artery‑disease cases are CKD (100 %)**; `no` is 59 % CKD / 41 % not‑CKD.\n\n**Why a count plot:** a rare positive class that is perfectly CKD — grouped counts highlight it; check that this is not a dataset artifact."),
    ("appet","**What the plot says:** all **82 `poor` appetite cases are CKD (100 %)**; `good` is 53 % CKD / 47 % not‑CKD.\n\n**Why a count plot:** binary symptom vs target → grouped bars; poor appetite is CKD‑exclusive in this sample."),
    ("pe","**What the plot says:** all **76 `yes` pedal‑edema cases are CKD (100 %)**; `no` is 54 % CKD / 46 % not‑CKD.\n\n**Why a count plot:** binary sign vs target → a grouped count plot; edema is a strong CKD indicator here."),
    ("ane","**What the plot says:** all **60 `yes` anaemia cases are CKD (100 %)**; `no` is 56 % CKD / 44 % not‑CKD.\n\n**Why a count plot:** binary sign vs target → grouped counts; anaemia is CKD‑linked and consistent with the low hemo/pcv signal."),
]
for col, text in cat_cells:
    md(f"### `{col}` (categorical) vs `classification` — count plot\n\n{text}")
    code(f'plot_count("{col}")')

# ── 4️⃣ Heatmaps
md("## 4. Categorical association heatmaps")
md("### Heatmap — categorical levels vs target")
md("**What the plot says:** every \"positive\" clinical flag (`htn=yes`, `dm=yes`, `cad=yes`, `pe=yes`, `ane=yes`, `appet=poor`, `pcc=present`, `ba=present`, `rbc=abnormal`, `pc=abnormal`) is **100 % CKD**, while the \"normal/absent\" levels hover near the 60/40 base rate.\n\n**Why a heatmap:** a row‑normalised cross‑tab puts every categorical level against the target in one view, making near‑separators and class skew immediately obvious.")
code("""rows, labels = [], []
for c in CAT_COLS:
    ct = pd.crosstab(clean[c], clean[TARGET], normalize="index").reindex(columns=ORDER).mul(100)
    for level, row in ct.iterrows():
        rows.append(row.values); labels.append(f"{c} = {level}")
level_target = pd.DataFrame(rows, index=labels, columns=[f"% {t}" for t in ORDER])
fig, ax = plt.subplots(figsize=(5, 8))
sns.heatmap(level_target, annot=True, fmt=".1f", cmap="RdBu_r", vmin=0, vmax=100,
            cbar_kws={"label": "% of level (row)"}, ax=ax)
ax.set_title("Categorical levels vs target"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.show()
level_target
""")

md("### Heatmap — Cramér's V between categorical features and target")
md("**What the plot says:** most clinical flags show strong association with the target, and many are also strongly associated with each other (they describe the same clinical picture).\n\n**Why a heatmap:** Cramér's V is the categorical analogue of correlation, so one matrix reveals redundancy / multicollinearity among the binary flags as well as their link to the target.")
code("""cols = CAT_COLS + [TARGET]
v_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
for i in cols:
    for j in cols:
        v_mat.loc[i, j] = 1.0 if i == j else cramers_v(clean[i], clean[j])

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(v_mat.astype(float), annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
            square=True, cbar_kws={"label": "Cramér's V"}, ax=ax)
ax.set_title("Cramér's V association — categorical features + target"); plt.tight_layout(); plt.show()
v_mat.astype(float).round(2)
""")

# ── 5️⃣ Summary tables
md("## 5. Feature–target summary")
code("""# Numeric summary
num_rows = []
for c in NUM_COLS:
    med = clean.groupby(TARGET)[c].median().reindex(ORDER)
    mean = clean.groupby(TARGET)[c].mean().reindex(ORDER)
    num_rows.append({
        "feature": c,
        "plot": PLOT_FOR[c],
        "notckd_median": round(med["notckd"], 3),
        "ckd_median": round(med["ckd"], 3),
        "median_diff": round(med["ckd"] - med["notckd"], 3),
        "notckd_mean": round(mean["notckd"], 3),
        "ckd_mean": round(mean["ckd"], 3),
        "null_%": round(clean[c].isna().mean() * 100, 1),
    })
num_summary = pd.DataFrame(num_rows).set_index("feature")
num_summary

# Categorical summary
cat_rows = []
for c in CAT_COLS:
    ct = pd.crosstab(clean[c], clean[TARGET], normalize="index").reindex(columns=ORDER)
    hi = ct["ckd"].idxmax()
    cat_rows.append({
        "feature": c,
        "level_most_linked_to_ckd": hi,
        "ckd_%_at_that_level": round(ct.loc[hi, "ckd"] * 100, 1),
        "n_at_that_level": int(clean[c].value_counts().get(hi, 0)),
        "null_%": round(clean[c].isna().mean() * 100, 1),
    })
cat_summary = pd.DataFrame(cat_rows).set_index("feature")
cat_summary
""")

md("## 6. Takeaways for modeling")
md("- **Strongest numeric signals:** `hemo`, `pcv`, `rc` (lower in CKD), `sg`/`al` (bar‑level shift); `sc`, `bu`, `bgr` also separate but are skewed and need scaling or tree models.\n- **Near‑perfect categorical separators:** every positive clinical flag (`htn`, `dm`, `cad`, `pe`, `ane`, `appet=poor`, `pcc`, `ba`, `rbc`, `pc`) is 100 % CKD here. **Validate on a hold‑out split** — these can inflate scores and may reflect selection bias.\n- **Missingness is informative (MNAR):** `rbc` 38 %, `rc` 33 %, `wc` 26 %, `pot` 22 %, `sod` 22 % → add `*_was_missing` flags (especially `rbc`, `pc`) instead of relying on mode fills alone.\n- **Multicollinearity:** `hemo` / `pcv` / `rc` are strongly inter‑correlated → keep one or regularise / use dimensionality reduction.\n- **Data errors:** `sod = 4.5`, `pot = 47` are almost certainly typos → fix or cap before modeling.\n- **Class imbalance (62.5 / 37.5):** use a stratified split and class weights / resampling.\n- **Drop `id`** — it is a sequential identifier with no signal.")

# ── 2️⃣ Execute notebook ─────────────────────────────────────────────────
# Remove old notebook if present (so we start clean)
if os.path.exists(NOTEBOOK_PATH):
    os.remove(NOTEBOOK_PATH)
if os.path.exists(BACKUP_PATH):
    os.remove(BACKUP_PATH)

nbformat.write(nb, NOTEBOOK_PATH)

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": PROJECT_ROOT}},
    allow_errors=False,
)
client.execute()
nbformat.write(nb, NOTEBOOK_PATH)

# ── 3️⃣ Verify ─────────────────────────────────────────────────────────────
v = nbf.read(NOTEBOOK_PATH, as_version=4)
ncode = sum(1 for c in v.cells if c.cell_type == "code")
nimg = sum(1 for c in v.cells if c.cell_type == "code"
           for o in c.get("outputs", [])
           if o.get("output_type") in ("display_data", "execute_result")
           and "image/png" in o.get("data", {}))
print(f"✅ Rebuilt notebook: {ncode} code cells, {nimg} executed plots")
# Keep a backup so you can cp it back if the editor overwrites again
import shutil
shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
print(f"💾 Backup saved to {BACKUP_PATH}")