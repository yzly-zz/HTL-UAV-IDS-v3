"""Few-Shot Transfer Learning 实验 — 国一关键证据
证明 HTL 冻结 Backbone + Domain Projector 在小样本下远超从头训练的基线。
用法: python scripts/few_shot_gcs.py --device cuda
"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import pandas as pd, numpy as np, os, sys, argparse, copy, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_engine.preprocessor import HeterogeneousDataPreprocessor, CSV_ENCODINGS
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import HTL_UAV_IDS

class VanillaMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,num_classes))
    def forward(self,x): return self.net(x)

class MobileNet1D(nn.Module):
    def __init__(self, input_dim, num_classes=2, shared_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, shared_dim)
        self.backbone = nn.Sequential(
            nn.Conv1d(1,16,3,padding=1,bias=False),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(16,16,3,padding=1,groups=16,bias=False),
            nn.Conv1d(16,32,1,bias=False),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,32,3,padding=1,groups=32,bias=False),
            nn.Conv1d(32,64,1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,num_classes))
    def forward(self,x):
        x=self.proj(x).unsqueeze(1)
        return self.classifier(self.backbone(x).squeeze(-1))


def safe_read_csv(fp):
    for e in CSV_ENCODINGS:
        try: return pd.read_csv(fp, low_memory=False, encoding=e)
        except (UnicodeDecodeError, UnicodeError): continue
    raise ValueError(f"Cannot decode: {fp}")

def train_one_model(model, train_ldr, test_ldr, epochs, lr, device, cw):
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_f1, best_st = 0.0, None
    for ep in range(epochs):
        model.train()
        for inp, lab in train_ldr:
            inp = inp.to(device, dtype=torch.float32)
            lab = lab.to(device, dtype=torch.long)
            opt.zero_grad()
            crit(model(inp), lab).backward()
            opt.step()
        model.eval()
        ap, al = [], []
        with torch.no_grad():
            for inp, lab in test_ldr:
                inp = inp.to(device, dtype=torch.float32)
                lab = lab.to(device, dtype=torch.long)
                _, pr = torch.max(model(inp), 1)
                ap.extend(pr.cpu().numpy())
                al.extend(lab.cpu().numpy())
        f1m = f1_score(al, ap, average="macro")
        if f1m > best_f1:
            best_f1 = f1m
            best_st = copy.deepcopy(model.state_dict())
    if best_st: model.load_state_dict(best_st)
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for inp, lab in test_ldr:
            inp = inp.to(device, dtype=torch.float32)
            lab = lab.to(device, dtype=torch.long)
            _, pr = torch.max(model(inp), 1)
            ap.extend(pr.cpu().numpy())
            al.extend(lab.cpu().numpy())
    ap = np.array(ap); al = np.array(al)
    return {
        "acc": (ap == al).mean(),
        "f1_macro": f1_score(al, ap, average="macro"),
        "f1_normal": f1_score(al, ap, pos_label=0),
        "f1_attack": f1_score(al, ap, pos_label=1),
    }

def run_experiment(device="cuda", seeds=(42, 123, 456, 789, 1024)):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    csv_path = "data/raw/UAV-NIDD/GSC Case3 Label.csv"
    base_model_path = "weights/source_base_model.pth"
    ratios = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
    epochs, lr = 20, 1e-4

    print("=" * 70)
    print("  Few-Shot Transfer Learning -- GCS Case3")
    print("  Device: {} | Ratios: {} | Runs: {}".format(device, ratios, len(seeds)))
    print("=" * 70)

    prep = HeterogeneousDataPreprocessor()
    df = safe_read_csv(csv_path)
    X_raw, y, feat_names = prep._clean_dataframe(df, label_col=None)
    input_dim = X_raw.shape[1]
    print("  Full: {:,} samples | Dim: {} | Attack: {:.2%}".format(len(X_raw), input_dim, y.mean()))

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y)
    prep.scaler.fit(X_train_full)
    X_test_s = prep.scaler.transform(X_test)
    test_ldr = DataLoader(IDSStreamDataset(X_test_s, y_test), batch_size=256, shuffle=False)

    cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train_full)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    print("  Class weights: Normal={:.3f}, Attack={:.3f}".format(cw[0], cw[1]))

    if not os.path.exists(base_model_path):
        print("  ERROR: {} not found!".format(base_model_path))
        return
    pretrained = torch.load(base_model_path, map_location=device)

    all_results = defaultdict(list)

    for ratio in ratios:
        n_samp = max(2, int(len(y_train_full) * ratio))
        n_normal = max(1, int(n_samp * (1 - y_train_full.mean())))
        n_attack = n_samp - n_normal
        print("")
        print("  " + "-" * 50)
        print("  Ratio: {:.0%} ({:,} samples: {}N + {}A)".format(ratio, n_samp, n_normal, n_attack))
        print("  " + "-" * 50)

        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)

            idx_n = np.where(y_train_full == 0)[0]
            idx_a = np.where(y_train_full == 1)[0]
            sn = np.random.choice(idx_n, min(n_normal, len(idx_n)), replace=False)
            sa = np.random.choice(idx_a, min(n_attack, len(idx_a)), replace=False)
            sidx = np.concatenate([sn, sa]); np.random.shuffle(sidx)

            X_tr = prep.scaler.transform(X_train_full[sidx])
            y_tr = y_train_full[sidx]
            tr_ldr = DataLoader(IDSStreamDataset(X_tr, y_tr), batch_size=min(64, len(y_tr)), shuffle=True)

            # --- HTL-UAV-IDS ---
            m_htl = HTL_UAV_IDS(input_dim=input_dim, shared_dim=64, num_classes=2).to(device)
            md = m_htl.state_dict()
            fd = {k: v for k, v in pretrained.items() if k in md and "projector" not in k}
            md.update(fd); m_htl.load_state_dict(md)
            m_htl.freeze_backbone_for_finetuning()
            r_htl = train_one_model(m_htl, tr_ldr, test_ldr, epochs, lr, device, cw_t)
            r_htl.update(ratio=ratio, n_samples=n_samp, seed=seed, model="HTL-UAV-IDS")
            all_results["HTL-UAV-IDS"].append(r_htl)

            # --- MLP ---
            m_mlp = VanillaMLP(input_dim=input_dim, num_classes=2).to(device)
            r_mlp = train_one_model(m_mlp, tr_ldr, test_ldr, epochs, lr, device, cw_t)
            r_mlp.update(ratio=ratio, n_samples=n_samp, seed=seed, model="MLP")
            all_results["MLP"].append(r_mlp)

            # --- MobileNet1D ---
            m_mbn = MobileNet1D(input_dim=input_dim, num_classes=2).to(device)
            r_mbn = train_one_model(m_mbn, tr_ldr, test_ldr, epochs, lr, device, cw_t)
            r_mbn.update(ratio=ratio, n_samples=n_samp, seed=seed, model="MobileNet1D")
            all_results["MobileNet1D"].append(r_mbn)

            print("    [seed={}] HTL F1={:.4f} | MLP F1={:.4f} | MBN F1={:.4f}".format(
                seed, r_htl["f1_macro"], r_mlp["f1_macro"], r_mbn["f1_macro"]))

    # --- Summary ---
    print("")
    print("=" * 70)
    print("  Few-Shot Results Summary")
    print("=" * 70)
    hdr = "{:<16s} {:>6s} {:>8s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
        "Model", "Ratio", "N", "Acc", "F1-Macro", "F1-Normal", "F1-Attack")
    print(hdr)
    print("-" * 75)

    summary_rows = []
    for model_name in ["HTL-UAV-IDS", "MLP", "MobileNet1D"]:
        for ratio in ratios:
            group = [r for r in all_results[model_name] if r["ratio"] == ratio]
            if not group: continue
            avg = {k: np.mean([g[k] for g in group]) for k in ["acc","f1_macro","f1_normal","f1_attack"]}
            std = {k: np.std([g[k] for g in group]) for k in ["acc","f1_macro","f1_normal","f1_attack"]}
            n = group[0]["n_samples"]
            print("{:<16s} {:>5.0%}  {:>6,d}  {:>7.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}".format(
                model_name, ratio, n, avg["acc"], avg["f1_macro"], avg["f1_normal"], avg["f1_attack"]))
            summary_rows.append({
                "model": model_name, "ratio": ratio, "n_samples": n,
                "acc_mean": avg["acc"], "acc_std": std["acc"],
                "f1_macro_mean": avg["f1_macro"], "f1_macro_std": std["f1_macro"],
                "f1_normal_mean": avg["f1_normal"], "f1_normal_std": std["f1_normal"],
                "f1_attack_mean": avg["f1_attack"], "f1_attack_std": std["f1_attack"],
            })

    os.makedirs("results", exist_ok=True)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/few_shot_gcs.csv", index=False, encoding="utf-8")
    print("  Saved: results/few_shot_gcs.csv")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {"HTL-UAV-IDS": "#2196F3", "MLP": "#FF9800", "MobileNet1D": "#4CAF50"}
        markers = {"HTL-UAV-IDS": "o", "MLP": "s", "MobileNet1D": "^"}

        for ax, metric, title in zip(axes,
                ["f1_macro", "f1_normal", "f1_attack"],
                ["F1-Macro", "F1-Normal (Key Metric)", "F1-Attack"]):
            for model_name in ["HTL-UAV-IDS", "MLP", "MobileNet1D"]:
                xs, ys, es = [], [], []
                for ratio in ratios:
                    group = [r for r in all_results[model_name] if r["ratio"] == ratio]
                    if not group: continue
                    xs.append(group[0]["n_samples"])
                    ys.append(np.mean([g[metric] for g in group]))
                    es.append(np.std([g[metric] for g in group]))
                ax.errorbar(xs, ys, yerr=es, marker=markers[model_name], color=colors[model_name],
                           label=model_name, capsize=4, linewidth=2, markersize=8)
            ax.set_xlabel("Training Samples", fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title("Few-Shot: " + title, fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/few_shot_learning_curves.png", dpi=150, bbox_inches="tight")
        print("  Saved: results/few_shot_learning_curves.png")
    except Exception as e:
        print("  Plot failed (non-critical): {}".format(e))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Transfer Learning Experiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()
    run_experiment(device=args.device, seeds=args.seeds)
