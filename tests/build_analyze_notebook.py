"""Generate tests/analyze_full_results.ipynb with robust, verbose agent analysis."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analyze_full_results.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [text]}


cells = []

cells.append(md("""# BAMBON benchmark — full results analysis (verbose)

This notebook analyses **one benchmark run** from `full_results.json`, optionally `full_negotiations.json` (trajectories), and `bambon_vs_all_categories.csv`.

### Single utility (violations penalised)

> **`utility`** for agent *i* = ODRL utility of the agreed contract, **0** if the contract breaches *i*'s own hard policy, **0** if no agreement. Averaged over **all** negotiations involving *i* (both provider and consumer roles).

`enforceable_rate` is reported separately as a compliance statistic, not as utility."""))

cells.append(md("## 0. Setup & helpers"))

cells.append(code(r'''import json, os, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
from IPython.display import display, Markdown

sns.set_theme(style="whitegrid", context="notebook")
mpl.rcParams.update({"figure.dpi":110,"savefig.dpi":130,"figure.facecolor":"white",
 "axes.titleweight":"bold","axes.titlesize":12,"axes.titlepad":10,"font.size":10,"legend.frameon":False})
BAMBON="#e4572e"; BASE="#3a6ea5"; ACCENT="#17a398"; MUTED="#9aa0a6"; GOOD="#2a9d4a"; BAD="#c0392b"
BAM3={"bambon_aggressive":"#b3261e","bambon_balanced":"#e4572e","bambon_exploratory":"#f3a26b"}
BAM_PROFILES=["bambon_aggressive","bambon_balanced","bambon_exploratory"]

def _can_style():
    try: pd.DataFrame({"a":[1]}).style.to_html(); return True
    except Exception: return False
HAS_STYLE=_can_style()

def _tests_root():
    cwd=Path.cwd()
    for d in (cwd, cwd/"tests"):
        if (d/"out_full").is_dir() or (d/"full_results.json").exists():
            return d
    return cwd

ROOT=str(_tests_root())
CANDIDATES=[
    os.path.join(ROOT,"out_full"), os.path.join(ROOT,"out_anac"), os.path.join(ROOT,"out_anl"),
    os.path.join(ROOT,"out_test"), os.path.join(ROOT,"out_anac_test"),
    "./out_full","./out_anac","./out_anl","./out_test","./out_anac_test","tests/out_full",".",
]

def find(name):
    for c in CANDIDATES:
        if not c or not os.path.isdir(c): continue
        for pat in (name, name.replace(".json","-*.json").replace(".csv","-*.csv")):
            hits=sorted(glob.glob(os.path.join(c,pat)))
            if hits:
                ex=[h for h in hits if os.path.basename(h)==name]
                return ex[0] if ex else hits[0]
    return None

def load_json_lenient(p):
    raw=open(p,"r",encoding="utf-8",errors="replace").read()
    return json.JSONDecoder().raw_decode(raw)[0]

def is_bam(s): return str(s).startswith(("bambon","Bayesian"))
def short(s): return str(s).replace("_provider","").replace("_consumer","")
def agent_color(a):
    s=short(a)
    if s in BAM3: return BAM3[s]
    return BAMBON if is_bam(a) else BASE

def present_in_index(df, names):
    return [n for n in names if n in df.index]

def safe_mean(series, default=float("nan")):
    return float(series.mean()) if len(series) else default

def _row_is_negotiation(row):
    return isinstance(row, dict) and {"provider","consumer","agreement"} <= set(row)

def _row_is_trajectory(row):
    return isinstance(row, dict) and "rounds" in row and "data" in row

def _extract_results(res):
    if isinstance(res, dict):
        recs = res.get("results") if "results" in res else None
        if recs and _row_is_negotiation(recs[0]): return res, recs
        return (res if any(k in res for k in ("overall","by_agent","scenario")) else None), None
    if isinstance(res, list) and res:
        if _row_is_negotiation(res[0]): return None, res
    return None, None

def normalize_results_df(df):
    if "odrl_utility_provider" not in df.columns and "utility_provider" in df.columns:
        df["odrl_utility_provider"]=df["utility_provider"]
    if "odrl_utility_consumer" not in df.columns and "utility_consumer" in df.columns:
        df["odrl_utility_consumer"]=df["utility_consumer"]
    for c in ("provider_hard_violations","consumer_hard_violations"):
        if c not in df.columns: df[c]=0
    for c in ("provider_violated","consumer_violated"):
        if c not in df.columns: df[c]=False
    if "contract_enforceable" not in df.columns:
        df["contract_enforceable"]=df.agreement & ~df.provider_violated & ~df.consumer_violated
    if "odrl_fairness" not in df.columns and "fairness" in df.columns:
        df["odrl_fairness"]=df["fairness"]
    return df

def find_benchmark_json():
    override=os.environ.get("BENCHMARK_DATA_DIR")
    dirs=[override] if override else []
    dirs += [c for c in CANDIDATES if c and c not in dirs]
    for c in dirs:
        if not c or not os.path.isdir(c): continue
        p=os.path.join(c,"full_results.json")
        if not os.path.isfile(p): continue
        try:
            res=load_json_lenient(p)
            agg, recs=_extract_results(res)
            if recs: return p, agg, recs, None
            if isinstance(res, list) and res and _row_is_trajectory(res[0]):
                return p, None, None, res
        except Exception: continue
    return None, None, None, None
'''))

cells.append(md("## 1. Load artifacts & run manifest"))

cells.append(code(r'''fr, AGG, recs, traj_misnamed = find_benchmark_json()
DATA_SOURCE = "full_results.json"
traj_from_fr = traj_misnamed
if recs is None:
    cv_path=find("bambon_vs_all_categories.csv")
    assert cv_path, "No valid full_results.json or bambon_vs_all_categories.csv found"
    df=normalize_results_df(pd.read_csv(cv_path))
    DATA_DIR=os.path.dirname(cv_path)
    DATA_SOURCE="bambon_vs_all_categories.csv (CSV fallback — violation columns may be absent)"
    fr=cv_path
else:
    df=normalize_results_df(pd.DataFrame(recs))
    DATA_DIR=os.path.dirname(fr)
ng_path=os.path.join(DATA_DIR,"full_negotiations.json")
if not os.path.isfile(ng_path):
    ng_path=find("full_negotiations.json")
negs=load_json_lenient(ng_path) if ng_path else (traj_from_fr or [])
cv_path=os.path.join(DATA_DIR,"bambon_vs_all_categories.csv")
csv_df=pd.read_csv(cv_path) if os.path.isfile(cv_path) else None

df["pcat"]=df.provider.map(short)
df["ccat"]=df.consumer.map(short)
df["fam"]=np.where(df.provider.map(is_bam),"BAMBON","baseline")
df["util_prov"]=np.where(df.agreement & (df.provider_hard_violations==0), df.odrl_utility_provider, 0.0)
df["util_cons"]=np.where(df.agreement & (df.consumer_hard_violations==0), df.odrl_utility_consumer, 0.0)
df["util_nash"]=df.util_prov*df.util_cons
HAS_TRAJ=len(negs)>0

agents=sorted(set(df.pcat)|set(df.ccat))
bam_agents=[a for a in agents if is_bam(a)]
base_agents=[a for a in agents if not is_bam(a)]
bam_profiles_present=[p for p in BAM_PROFILES if p in agents]

manifest=pd.DataFrame({
    "item":["data_dir","results_source","full_results","trajectories","csv","negotiations","agents","bambon_agents","baseline_agents","bambon_profiles","has_trajectories"],
    "value":[DATA_DIR, DATA_SOURCE, fr, ng_path or ("(misnamed full_results.json)" if traj_from_fr else "(none)"),
             cv_path or "(none)", len(df), len(agents), len(bam_agents), len(base_agents),
             ", ".join(bam_profiles_present) or "(none)", HAS_TRAJ]})
display(manifest.style.hide(axis="index").set_caption("Run manifest")) if HAS_STYLE else display(manifest)

print("Utility sanity — non-agreement:", round(safe_mean(df[~df.agreement].util_prov, 0),3),
      "| own-violation:", round(safe_mean(df[df.provider_violated].util_prov, 0),3) if "provider_violated" in df.columns else "n/a",
      "| clean-deal:", round(safe_mean(df[df.agreement & ~df.provider_violated].util_prov if "provider_violated" in df.columns else df[df.agreement].util_prov),3))
'''))

cells.append(md("## 2. Headline KPIs"))

cells.append(code(r'''ov=AGG.get("overall",{}) if AGG else {}
nb=ov.get("nash_bargaining",{}) if ov else {}
kpi=pd.DataFrame({"metric":[
    "scenario","negotiations","runs / pairing","pairings","agreement rate","enforceable-contract rate",
    "mean utility (single, both sides)","mean Nash (utility product)","Jain fairness (mean)",
    "mean turns (agreed)","mean wall_time_s"],
  "value":[
    AGG.get("scenario","?") if AGG else "?",
    AGG.get("n_negotiations",len(df)) if AGG else len(df),
    AGG.get("n_runs_per_pairing","?") if AGG else "?",
    AGG.get("n_pairings","?") if AGG else "?",
    f"{df.agreement.mean():.2%}",
    f"{df.contract_enforceable.mean():.2%}",
    f"{pd.concat([df.util_prov,df.util_cons]).mean():.3f}",
    f"{df.util_nash.mean():.3f}",
    f"{nb.get('jains_fairness_index_mean',float('nan')):.3f}" if nb else "n/a",
    f"{df[df.agreement].turns.mean():.1f}" if 'turns' in df.columns else "n/a",
    f"{df.wall_time_s.mean():.3f}" if 'wall_time_s' in df.columns else "n/a",
  ]})
display(kpi.style.hide(axis="index").set_caption("Run headline")) if HAS_STYLE else display(kpi)
'''))

cells.append(md("## 3. Agent roster (every agent in this run)"))

cells.append(code(r'''def agent_table(df):
    rows=[]
    for a in sorted(set(df.pcat)|set(df.ccat)):
        gp=df[df.pcat==a]; gc=df[df.ccat==a]
        util=pd.concat([gp.util_prov, gc.util_cons])
        nash=pd.concat([gp.util_nash, gc.util_nash])
        agr=pd.concat([gp.agreement, gc.agreement])
        enf=pd.concat([gp.contract_enforceable, gc.contract_enforceable])
        fair=pd.concat([gp[gp.agreement].odrl_fairness, gc[gc.agreement].odrl_fairness]) if "odrl_fairness" in df.columns else pd.Series(dtype=float)
        viol=int(gp.provider_violated.sum()+gc.consumer_violated.sum()) if "provider_violated" in df.columns else 0
        rows.append([a, is_bam(a+"_provider"), len(gp), len(gc), len(gp)+len(gc),
                     agr.mean(), enf.mean(), util.mean(), nash.mean(),
                     fair.mean() if len(fair) else np.nan, viol])
    return pd.DataFrame(rows,columns=[
        "agent","is_bambon","n_as_provider","n_as_consumer","n_total",
        "agreement_rate","enforceable_rate","utility","nash","fairness","violations"]).set_index("agent")

AT=agent_table(df).sort_values("utility",ascending=False)
AT["utility_rank"]=range(1,len(AT)+1)
cols=["is_bambon","n_as_provider","n_as_consumer","n_total","agreement_rate","enforceable_rate",
      "utility","nash","fairness","violations","utility_rank"]
if HAS_STYLE:
    display(AT[cols].style.background_gradient(subset=["enforceable_rate","utility","nash"],cmap="YlGn")
            .background_gradient(subset=["violations"],cmap="Reds")
            .format({c:"{:.3f}" for c in ["agreement_rate","enforceable_rate","utility","nash","fairness"]})
            .format({"violations":"{:.0f}","n_as_provider":"{:.0f}","n_as_consumer":"{:.0f}","n_total":"{:.0f}","utility_rank":"{:.0f}"})
            .set_caption("Per-agent scorecard (single utility, sorted by utility)"))
else:
    display(AT[cols].round(3))

# BAMBON summary (only agents present in this run)
bam=AT[AT.is_bambon]
base=AT[~AT.is_bambon]
if len(bam):
    ref=bam.index[0]
    print(f"BAMBON agents in run: {list(bam.index)}")
    print(f"  Best BAMBON by utility: {bam.index[0]} ({bam.utility.iloc[0]:.3f})")
    if len(base):
        print(f"  vs best baseline {base.index[0]} ({base.utility.iloc[0]:.3f})")
    print(f"  BAMBON enforceable_rate mean: {bam.enforceable_rate.mean():.0%} | baseline: {base.enforceable_rate.mean():.0%}" if len(base) else
          f"  BAMBON enforceable_rate mean: {bam.enforceable_rate.mean():.0%}")
    print(f"  BAMBON violations total: {int(bam.violations.sum())} | baseline: {int(base.violations.sum())}" if len(base) else "")
else:
    print("No BAMBON agents in this run.")
'''))

cells.append(md("## 4. Role split — provider vs consumer (per agent)"))

cells.append(code(r'''def role_split(df, agent):
    gp=df[df.pcat==agent]; gc=df[df.ccat==agent]
    return pd.DataFrame({
        "role":["provider","consumer"],
        "n":[len(gp),len(gc)],
        "agreement_rate":[gp.agreement.mean(), gc.agreement.mean()],
        "enforceable_rate":[gp.contract_enforceable.mean(), gc.contract_enforceable.mean()],
        "utility":[gp.util_prov.mean(), gc.util_cons.mean()],
        "odrl_raw":[gp.odrl_utility_provider.mean() if len(gp) else np.nan,
                    gc.odrl_utility_consumer.mean() if len(gc) else np.nan],
        "violations":[int(gp.provider_violated.sum()) if "provider_violated" in gp.columns else 0,
                      int(gc.consumer_violated.sum()) if "consumer_violated" in gc.columns else 0],
    }).set_index("role")

for a in AT.index:
    print(f"\n### {a}")
    display(role_split(df,a).round(3))
'''))

cells.append(md("## 5. Pairing heatmaps (agreement, enforceable, utility)"))

cells.append(code(r'''metrics=[("agreement","Agreement rate"),("contract_enforceable","Enforceable rate"),
           ("util_prov","Provider utility (single)"),("util_cons","Consumer utility (single)"),("util_nash","Nash product")]
fig,axes=plt.subplots(2,3,figsize=(16,9))
for ax,(col,title) in zip(axes.ravel(),metrics):
    if col not in df.columns: ax.set_visible(False); continue
    piv=df.pivot_table(index="pcat",columns="ccat",values=col,aggfunc="mean")
    sns.heatmap(piv,annot=True,fmt=".2f",cmap="RdYlGn",vmin=0,vmax=1 if col!="util_nash" else None,
                linewidths=.5,linecolor="white",cbar_kws={"label":col},ax=ax)
    ax.set(title=title,xlabel="consumer",ylabel="provider")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=35,ha="right")
plt.suptitle("Pairing heatmaps",fontweight="bold",y=1.02)
plt.tight_layout(); plt.show()
'''))

cells.append(md("## 6. Violation & compliance forensics"))

cells.append(code(r'''vc=pd.DataFrame({
    "metric":["agreements","provider own-violation deals","consumer own-violation deals","no agreement",
              "clean deals (agree, no viol)","enforceable contracts"],
    "count":[int(df.agreement.sum()),
             int((df.agreement & df.provider_violated).sum()) if "provider_violated" in df.columns else 0,
             int((df.agreement & df.consumer_violated).sum()) if "consumer_violated" in df.columns else 0,
             int((~df.agreement).sum()),
             int((df.agreement & ~df.provider_violated & ~df.consumer_violated).sum()) if "provider_violated" in df.columns else int(df.agreement.sum()),
             int(df.contract_enforceable.sum()) if "contract_enforceable" in df.columns else 0],
})
display(vc)

if "provider_violated" in df.columns:
    viol_agents=AT[AT.violations>0].sort_values("violations",ascending=False)
    if len(viol_agents):
        print("Agents with own-policy violations:")
        display(viol_agents[["violations","agreement_rate","enforceable_rate","utility"]])
    else:
        print("No own-policy violations in this run.")

order=AT.sort_values("enforceable_rate",ascending=False).index
fig,axes=plt.subplots(1,2,figsize=(13,4.6))
ax=axes[0]; y=np.arange(len(order)); h=0.38
ax.barh(y+h/2, AT.loc[order,"agreement_rate"], h, color=MUTED, label="agreement rate")
ax.barh(y-h/2, AT.loc[order,"enforceable_rate"], h,
        color=[agent_color(a) for a in order], label="enforceable rate")
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
ax.set(title="Agreement vs enforceable rate",xlim=(0,1.05)); ax.legend(fontsize=8,loc="lower right")
ax=axes[1]; v=AT.loc[order,"violations"].astype(float)
ax.barh(range(len(order)),v,color=[GOOD if x==0 else BAD for x in v],edgecolor="white")
for i,x in enumerate(v):
    if max(v)>0: ax.text(x+max(v)*.01+0.1,i,f"{int(x)}",va="center",fontsize=9)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
ax.set(title="Own hard-policy violations",xlabel="count")
plt.tight_layout(); plt.show()
'''))

cells.append(md("## 7. Family comparison (BAMBON vs baseline vs other)"))

cells.append(code(r'''def agent_family(a):
    if is_bam(a): return "BAMBON"
    if a.lower().endswith("agent") or "anl" in a.lower(): return "ANL/other"
    return "baseline"

fam_rows=[]
for fam in sorted(set(agent_family(a) for a in agents)):
    sub=AT[[agent_family(a)==fam for a in AT.index]]
    fam_rows.append([fam,len(sub),sub.agreement_rate.mean(),sub.enforceable_rate.mean(),
                     sub.utility.mean(),sub.nash.mean(),int(sub.violations.sum())])
fam_df=pd.DataFrame(fam_rows,columns=["family","n_agents","agreement_rate","enforceable_rate","utility","nash","violations"])
display(fam_df.round(3).style.set_caption("Pooled by agent family")) if HAS_STYLE else display(fam_df.round(3))

# Per pairing category summary
pair=df.groupby(["pcat","ccat"]).agg(
    n=("agreement","count"), agreement=("agreement","mean"),
    util_prov=("util_prov","mean"), util_cons=("util_cons","mean"),
    turns=("turns","mean") if "turns" in df.columns else ("agreement","count"),
).reset_index()
pair["provider_fam"]=pair.pcat.map(agent_family)
pair["consumer_fam"]=pair.ccat.map(agent_family)
display(pair.sort_values("util_prov",ascending=False).head(20).round(3))
'''))

cells.append(md("## 8. Outcome space by provider (single utility)"))

cells.append(code(r'''provs=sorted(df.pcat.unique(), key=lambda x:(not x.startswith("bambon"),x))
ncols=min(4,max(1,len(provs)))
nrows=int(np.ceil(len(provs)/ncols))
fig,axes=plt.subplots(nrows,ncols,figsize=(3.8*ncols,3.5*nrows),sharex=True,sharey=True)
axes=np.atleast_1d(axes).ravel()
for i,p in enumerate(provs):
    ax=axes[i]; s=df[df.pcat==p]
    c=agent_color(p)
    ax.scatter(s.util_prov,s.util_cons,s=8,alpha=.35,color=c,edgecolor="none")
    ax.plot([0,1],[0,1],ls="--",color=MUTED,lw=.8)
    ax.set_title(f"{p} (n={len(s)})",fontsize=9.5); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    if i%ncols==0: ax.set_ylabel("consumer utility")
    if i//ncols>=nrows-1: ax.set_xlabel("provider utility")
for j in range(len(provs),len(axes)): axes[j].set_visible(False)
fig.suptitle("Outcome space by provider — void/failed deals at origin",fontweight="bold",y=1.0)
plt.tight_layout(); plt.show()
'''))

cells.append(md("## 9. Trajectory analysis (requires `full_negotiations.json`)"))

cells.append(code(r'''if not HAS_TRAJ:
    display(Markdown("**No trajectories loaded** — skip sections 9–10. Re-run benchmark with trajectory export enabled (e.g. `out_full`)."))
else:
    def setify(v):
        if v is None: return set()
        return set(map(str,v)) if isinstance(v,(list,tuple)) else {str(v)}
    def jac(a,b):
        a,b=set(a),set(b); return (len(a&b)/len(a|b)) if (a or b) else 1.0
    def finalv(t,k):
        vv=[v for v in t["data"][k] if v is not None]; return vv[-1] if vv else None
    PID={"permission:use:fileFormat":{"JSON"},"permission:use:industry":{"research"},
         "permission:use:purpose":{"research","analytics","open-source"},"permission:use:language":{"en","fr","gr"}}
    CID={"permission:use:fileFormat":{"XML"},"permission:use:industry":{"research","production"},
         "permission:use:purpose":{"analytics","publication"},"permission:use:language":{"en"}}
    sample=next((t for t in negs if t.get("data")), None)
    ENUM=[k for k in PID if sample and k in sample.get("data",{})]
    LAB={k:k.split(":")[-1] for k in ENUM}
    print(f"Trajectory negotiations: {len(negs)} | agreed: {sum(1 for t in negs if t.get('agreed'))} | enum issues: {len(ENUM)}")

    rows=[]
    for t in negs:
        if not t.get("agreed"): continue
        fam="BAMBON" if is_bam(t["provider"]) else "baseline"
        for k in ENUM:
            fv=setify(finalv(t,k)); rows.append((fam,LAB[k],jac(fv,PID[k]),jac(fv,CID[k])))
    if rows:
        gd=pd.DataFrame(rows,columns=["fam","issue","ov_prov","ov_cons"]).groupby(["fam","issue"]).mean().reset_index()
        fig,axes=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
        for ax,fam in zip(axes,["BAMBON","baseline"]):
            sub=gd[gd.fam==fam]
            if sub.empty:
                ax.set_title(f"{fam} as provider (no data)"); continue
            s=sub.set_index("issue").reindex([LAB[k] for k in ENUM if k in ENUM])
            x=np.arange(len(s)); w=0.38
            ax.bar(x-w/2,s.ov_prov,w,color=BASE,label="overlap w/ PROVIDER ideal")
            ax.bar(x+w/2,s.ov_cons,w,color=BAMBON,label="overlap w/ CONSUMER ideal")
            ax.set_xticks(x); ax.set_xticklabels(s.index,rotation=20); ax.set_title(f"{fam} as provider"); ax.set_ylim(0,1.05)
            if fam=="BAMBON": ax.set_ylabel("mean Jaccard overlap"); ax.legend(fontsize=8)
        fig.suptitle("Categorical/set issues — overlap with ideals",fontweight="bold",y=1.02)
        plt.tight_layout(); plt.show()
'''))

cells.append(code(r'''if HAS_TRAJ:
    def parity(rr): return "prov" if (rr==-1 or rr%2==1) else "cons"
    def jac_path(t,k,nbins=20):
        last={"prov":None,"cons":None}; xs=[]; gs=[]
        for rr,vv in zip(t["rounds"],t["data"][k]):
            if vv is None: continue
            last[parity(rr)]=setify(vv)
            if last["prov"] is not None and last["cons"] is not None:
                xs.append(rr); gs.append(jac(last["prov"],last["cons"]))
        if len(gs)<2: return None
        xs=np.array(xs,float); xs=(xs-xs.min())/(xs.max()-xs.min()+1e-9)
        return np.interp(np.linspace(0,1,nbins),xs,gs)
    if ENUM:
        fig,ax=plt.subplots(figsize=(7.6,4.6))
        for k in ENUM:
            paths=[p for p in (jac_path(t,k) for t in negs if t.get("agreed")) if p is not None]
            if paths: ax.plot(np.linspace(0,1,20),np.mean(paths,axis=0),lw=1.9,label=LAB[k])
        ax.set(title="Set/enum convergence (Jaccard prov vs cons)",xlabel="negotiation progress",ylabel="Jaccard",ylim=(0,1.02))
        ax.legend(fontsize=8); plt.tight_layout(); plt.show()

    def prov_path(t,issue):
        pv=[]
        for rr,vv in zip(t["rounds"],t["data"][issue]):
            if vv is None: continue
            if rr==-1 or (rr>0 and rr%2==1): pv.append(vv)
        return pv
    def mean_path(prov,opp,issue,nmoves=8):
        ser=[prov_path(t,issue) for t in negs if t["provider"]==prov and t["consumer"]==opp and t.get("agreed")]
        ser=[s for s in ser if len(s)>=3]
        if not ser: return None
        m=min(min(len(s) for s in ser),nmoves); return np.mean([s[:m] for s in ser],axis=0)
    issue="permission:use:percentage"
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    ax=axes[0]
    for prof in bam_profiles_present:
        col=BAM3.get(prof,BAMBON)
        cons=sorted(set(t["consumer"] for t in negs if t["provider"]==prof+"_provider"))
        for opp in cons[:3]:
            m=mean_path(prof+"_provider",opp,issue)
            if m is not None: ax.plot(range(len(m)),m,"-o",ms=4,color=col,label=f"{short(prof)} vs {short(opp)}")
    ax.set(title="(a) BAMBON profile concession paths",xlabel="provider move #",ylabel="offered value"); ax.legend(fontsize=8)
    ax=axes[1]
    opps=sorted(set(t["consumer"] for t in negs if t["provider"].startswith("bambon")))
    for opp in opps[:8]:
        m=mean_path("bambon_balanced_provider",opp,issue,nmoves=12) if "bambon_balanced_provider" in {t["provider"] for t in negs} else None
        if m is not None: ax.plot(range(len(m)),m,"-o",ms=3,label=short(opp))
    ax.set(title="(b) BAMBON vs opponents on percentage",xlabel="provider move #",ylabel="offered value"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.show()

    ISS=list(sample["data"].keys()) if sample else []
    def itype(k):
        for t in negs:
            if k in t.get("data",{}):
                v=[x for x in t["data"][k] if x is not None]
                if v: return "numeric" if isinstance(v[-1],(int,float)) and not isinstance(v[-1],bool) else "categorical/temporal"
        return "unknown"
    ITYPE={k:itype(k) for k in ISS}
    def settle(t,issue):
        s=[(i,v) for i,v in zip(t["rounds"],t["data"][issue]) if v is not None]
        if not s: return None
        fin=str(s[-1][1]); k=len(s)-1
        while k>0 and str(s[k-1][1])==fin: k-=1
        return s[k][0]
    rows=[]
    for t in negs:
        for k in ISS:
            r=settle(t,k)
            if r is not None and r>=0: rows.append((ITYPE[k],r,short(t["provider"]),short(t["consumer"])))
    if rows:
        sd=pd.DataFrame(rows,columns=["type","settle_round","provider","consumer"])
        fig,axes=plt.subplots(1,2,figsize=(13,4.4))
        ax=axes[0]
        for typ,col in [("numeric",ACCENT),("categorical/temporal","#c44")]:
            sub=sd[sd.type==typ]
            if len(sub): ax.hist(sub.settle_round,bins=range(0,120,4),alpha=.6,color=col,label=typ)
        ax.set(title="Issue freeze round by type",xlabel="round",ylabel="count"); ax.legend(fontsize=8)
        ax=axes[1]
        rl=[len(t["rounds"]) for t in negs]
        ax.hist(rl,bins=range(0,max(rl)+4,3),color=BASE,edgecolor="white",alpha=.9)
        ax.axvline(np.median(rl),color=BAMBON,ls="--",lw=2,label=f"median {int(np.median(rl))}")
        ax.set(title="Negotiation length (rounds)",xlabel="rounds"); ax.legend()
        plt.tight_layout(); plt.show()
'''))

cells.append(md("## 10. BAMBON profile differentiation (only profiles present in run)"))

cells.append(code(r'''bp=AT[AT.is_bambon]
M=["agreement_rate","enforceable_rate","utility","nash","fairness"]
profiles_present=[p for p in BAM_PROFILES if p in bp.index]
if len(profiles_present)>=2:
    dd=pd.DataFrame({short(i):bp.loc[i,M] for i in profiles_present})
    dd["spread"]=dd.max(axis=1)-dd.min(axis=1)
    cap="Inter-profile differences"
    display(dd.style.background_gradient(subset=["spread"],cmap="Reds").format("{:.4f}").set_caption(cap)) if HAS_STYLE else display(dd.round(4))
    print(f"Max spread across metrics: {dd['spread'].max():.4f}")
elif len(profiles_present)==1:
    print(f"Only one BAMBON profile in run: {profiles_present[0]} — profile comparison N/A.")
else:
    print("No BAMBON profiles in agent table.")
'''))

cells.append(md("## 11. CSV cross-check (if exported)"))

cells.append(code(r'''if csv_df is not None:
    print(f"CSV rows: {len(csv_df)} | results rows: {len(df)} | match: {len(csv_df)==len(df)}")
    if len(csv_df)==len(df):
        extra=[c for c in csv_df.columns if c not in df.columns]
        print("CSV-only columns:", extra[:15], "..." if len(extra)>15 else "")
    display(csv_df.groupby(csv_df.provider.map(short)).agg(
        agreement=("agreement","mean"),
        utility_p=("utility_provider","mean") if "utility_provider" in csv_df.columns else ("agreement","count"),
    ).round(3).head(20))
else:
    print("No bambon_vs_all_categories.csv in data dir.")
'''))

cells.append(md("## 12. Dynamic takeaways (from this run only)"))

cells.append(code(r'''bam=AT[AT.is_bambon] if len(AT[AT.is_bambon]) else pd.DataFrame()
base=AT[~AT.is_bambon] if len(AT[~AT.is_bambon]) else pd.DataFrame()
lines=[]
lines.append(f"Results source: {DATA_SOURCE}.")
lines.append(f"Data: {DATA_DIR} — {len(df)} negotiations, {len(agents)} agents ({len(bam_agents)} BAMBON, {len(base_agents)} baseline).")
lines.append(f"Agreement {df.agreement.mean():.1%}, enforceable contracts {df.contract_enforceable.mean():.1%}.")
if "provider_violated" in df.columns:
    _ov=safe_mean(df[df.provider_violated].util_prov, 0)
    _cd=safe_mean(df[df.agreement & ~df.provider_violated].util_prov)
else:
    _ov=0.0
    _cd=safe_mean(df[df.agreement].util_prov)
lines.append(f"Single utility: non-agreement={safe_mean(df[~df.agreement].util_prov, 0):.3f}, own-violation={_ov:.3f}, clean-deal={_cd:.3f}.")
if len(bam):
    lines.append(f"BAMBON ({len(bam)} agents): violations={int(bam.violations.sum())}, enforceable={bam.enforceable_rate.mean():.0%}, utility={bam.utility.mean():.3f}, Nash={bam.nash.mean():.3f}.")
    lines.append(f"  Profiles present: {', '.join(bam_profiles_present) or 'none'}.")
if len(base):
    lines.append(f"Baselines ({len(base)} agents): violations={int(base.violations.sum())}, enforceable={base.enforceable_rate.mean():.0%}, utility={base.utility.mean():.3f}, best={base.index[0]} ({base.utility.iloc[0]:.3f}).")
if len(bam) and len(base):
    rk=list(AT.index)
    best_bam=bam.index[0]
    lines.append(f"Utility rank of top BAMBON ({best_bam}): #{rk.index(best_bam)+1}/{len(rk)}.")
if HAS_TRAJ:
    lines.append(f"Trajectories: {len(negs)} negotiations with round-by-round issue data.")
else:
    lines.append("Trajectories: not available — enable full_negotiations.json export for concession-path analysis.")
for L in lines:
    print("-", L)
'''))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
