"""Statistical analysis for the original epsilon/resolution experiments and transferability results."""
import math, csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "datasets"
TRANSFER = ROOT / "results" / "transferability_results.csv"
OUT = ROOT / "results" / "statistical_analysis.csv"


def prop_z_test(x1, n1, x2, n2):
    p1, p2 = x1/n1, x2/n2
    p = (x1+x2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1+1/n2))
    z = 0.0 if se == 0 else (p1-p2)/se
    # two-sided normal approximation
    pval = math.erfc(abs(z)/math.sqrt(2))
    return p1, p2, z, pval


def analyze_file(path, label):
    df = pd.read_csv(path)
    df["changed"] = df["original_class"] != df["cloaked_class"]
    out=[]
    for method, g in df.groupby("method"):
        changed=int(g["changed"].sum()); n=len(g)
        out.append({"comparison":label,"method":method,"n":n,"changed":changed,"rate":changed/n})
    return out


def main():
    rows = analyze_file(RESULTS/"results.csv", "default_epsilon") + analyze_file(RESULTS/"results_epsilon8.csv", "epsilon8")
    a=pd.read_csv(RESULTS/"results.csv"); b=pd.read_csv(RESULTS/"results_epsilon8.csv")
    for method in sorted(set(a.method)):
        x1=int(((a.method==method)&(a.original_class!=a.cloaked_class)).sum()); n1=int((a.method==method).sum())
        x2=int(((b.method==method)&(b.original_class!=b.cloaked_class)).sum()); n2=int((b.method==method).sum())
        p1,p2,z,pv=prop_z_test(x1,n1,x2,n2)
        rows.append({"comparison":f"default_vs_epsilon8_{method}","method":method,"n":n1+n2,"changed":x1+x2,"rate":p2-p1,"z_stat":z,"p_value":pv})
    if TRANSFER.exists():
        t=pd.read_csv(TRANSFER)
        for (s,tgt),g in t.groupby(["source_model","target_model"]):
            changed=int(g["prediction_changed"].sum()); n=len(g)
            rows.append({"comparison":f"transfer_{s}_to_{tgt}","method":"fgsm","n":n,"changed":changed,"rate":changed/n})
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\\nSaved: {OUT}")

if __name__ == "__main__": main()
