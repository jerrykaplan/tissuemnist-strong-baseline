"""
Multi-seed retraining of ResNet-18 + full_aug on TissueMNIST,
with AUC computation and weight saving.

Retrains the 4 seeds (1, 7, 13, 42) and computes one-vs-rest AUC alongside
accuracy. Saves test set softmax scores so additional metrics can be computed
later without retraining.

Reference: published TissueMNIST 28x28 baselines report AUC values:
  ResNet-18 (28): AUC 0.930, ACC 0.676
  ResNet-50 (28): AUC 0.931, ACC 0.680
  AutoKeras:      AUC 0.941, ACC 0.703

Our prior accuracy result: 0.7342 +/- 0.0007 across seeds 1, 7, 13.
"""
import os, sys, time, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ResNet18_28, TissueDS

torch.set_num_threads(8)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NPZ_PATH    = '/workspace/ahd_tissue/tissuemnist.npz'
RESULTS_DIR = '/workspace/ahd_tissue/cnn_results'
ALL_CLASSES = list(range(8))
N_CLASSES = 8


def evaluate(model, loader):
    """Run model on loader, return (predictions, true labels, softmax scores)."""
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x)
            scores = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_scores))


def compute_metrics(preds, labels, scores):
    """Compute accuracy, per-class accuracy, per-class AUC, and macro AUC."""
    accuracy = (preds == labels).mean()

    per_class_acc = np.zeros(N_CLASSES)
    per_class_auc = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        m = (labels == c)
        per_class_acc[c] = ((preds == labels) & m).sum() / max(m.sum(), 1)
        # Binary AUC: this class vs all others, using softmax probability of this class
        binary_labels = m.astype(int)
        per_class_auc[c] = roc_auc_score(binary_labels, scores[:, c])

    macro_auc = per_class_auc.mean()
    # sklearn macro_ovr should give same answer:
    sklearn_macro_auc = roc_auc_score(labels, scores, multi_class='ovr', average='macro')

    return {
        'accuracy': accuracy,
        'per_class_acc': per_class_acc,
        'per_class_auc': per_class_auc,
        'macro_auc': macro_auc,
        'sklearn_macro_auc': sklearn_macro_auc,
    }


def train_one_seed(seed, epochs=40, lr=1e-3, batch_size=128):
    """Train a single seed, save weights, return metrics + test scores."""
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    print(f"\n=== Seed {seed} ===")

    npz = np.load(NPZ_PATH)
    train_X = npz['train_X']; train_y = npz['train_y']
    val_X = npz['val_X']; val_y = npz['val_y']
    test_X = npz['test_X']; test_y = npz['test_y']

    train_ds = TissueDS(train_X, train_y, augment=True)
    val_ds   = TissueDS(val_X,   val_y,   augment=False)
    test_ds  = TissueDS(test_X,  test_y,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2)

    model = ResNet18_28(num_classes=N_CLASSES, in_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs // 2, 3 * epochs // 4], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0; n_seen = 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        scheduler.step()
        train_loss = total_loss / n_seen

        # Quick val check
        model.eval()
        v_correct = 0; v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        val_acc = v_correct / v_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:>2d}/{epochs}: loss={train_loss:.4f}  "
                  f"val={val_acc:.4f}  ({elapsed:.1f}s)")

    # Final evaluation with the final epoch's model (matches our prior reporting)
    print(f"  Computing final metrics...")
    preds, labels, scores = evaluate(model, test_loader)
    final_metrics = compute_metrics(preds, labels, scores)

    # Also evaluate at best-val checkpoint
    bv_model = ResNet18_28(num_classes=N_CLASSES, in_channels=1).to(DEVICE)
    bv_model.load_state_dict(best_state)
    bv_preds, bv_labels, bv_scores = evaluate(bv_model, test_loader)
    bv_metrics = compute_metrics(bv_preds, bv_labels, bv_scores)

    # Save final-epoch weights for later use
    weights_path = os.path.join(RESULTS_DIR, f'auc_seed{seed}.pth')
    torch.save(model.state_dict(), weights_path)

    print(f"  Final test acc: {final_metrics['accuracy']:.4f}, "
          f"AUC: {final_metrics['macro_auc']:.4f}")
    print(f"  Per-class AUC: " +
          ' '.join(f'{v:.3f}' for v in final_metrics['per_class_auc']))

    return {
        'seed': seed,
        'final': final_metrics,
        'best_val': bv_metrics,
        'best_val_acc': best_val_acc,
        'test_scores_final': scores,
        'test_scores_bestval': bv_scores,
        'test_labels': labels,
    }


def main(seeds, epochs):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    overall_start = time.time()
    for seed in seeds:
        t0 = time.time()
        result = train_one_seed(seed, epochs=epochs)
        result['wall_time'] = time.time() - t0
        results.append(result)
        print(f"  Wall time: {result['wall_time']/60:.1f} min")

    out_path = os.path.join(RESULTS_DIR, 'auc_multiseed.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    print(f"\nSaved to {out_path}")

    # Summary
    final_accs = np.array([r['final']['accuracy'] for r in results])
    final_aucs = np.array([r['final']['macro_auc'] for r in results])
    bv_accs = np.array([r['best_val']['accuracy'] for r in results])
    bv_aucs = np.array([r['best_val']['macro_auc'] for r in results])

    print(f"\n{'='*70}")
    print(f"MULTI-SEED ACCURACY AND AUC SUMMARY")
    print(f"{'='*70}")
    print(f"  Seeds: {[r['seed'] for r in results]}")
    print(f"")
    print(f"  FINAL EPOCH:")
    print(f"    Accuracy:  " + ' '.join(f'{a:.4f}' for a in final_accs) +
          f"  -> mean {final_accs.mean():.4f} ± {final_accs.std():.4f}")
    print(f"    Macro AUC: " + ' '.join(f'{a:.4f}' for a in final_aucs) +
          f"  -> mean {final_aucs.mean():.4f} ± {final_aucs.std():.4f}")
    print(f"")
    print(f"  BEST-VAL EPOCH:")
    print(f"    Accuracy:  " + ' '.join(f'{a:.4f}' for a in bv_accs) +
          f"  -> mean {bv_accs.mean():.4f} ± {bv_accs.std():.4f}")
    print(f"    Macro AUC: " + ' '.join(f'{a:.4f}' for a in bv_aucs) +
          f"  -> mean {bv_aucs.mean():.4f} ± {bv_aucs.std():.4f}")
    print(f"")

    # Per-class AUC averages
    per_class_aucs = np.array([r['final']['per_class_auc'] for r in results])
    pc_mean = per_class_aucs.mean(axis=0)
    pc_std = per_class_aucs.std(axis=0)
    print(f"  Per-class AUC (mean ± std across seeds):")
    for c in range(N_CLASSES):
        print(f"    Class {c}: {pc_mean[c]:.4f} ± {pc_std[c]:.4f}")

    print(f"")
    print(f"  COMPARISON TO PUBLISHED:")
    print(f"    ResNet-18 (28) official:  ACC 0.676  AUC 0.930")
    print(f"    ResNet-50 (28) official:  ACC 0.680  AUC 0.931")
    print(f"    AutoKeras:                ACC 0.703  AUC 0.941")
    print(f"    Ours (mean):              ACC {final_accs.mean():.4f}  AUC {final_aucs.mean():.4f}")
    print(f"    Δ vs AutoKeras AUC:       {final_aucs.mean() - 0.941:+.4f}")
    print(f"")
    print(f"  Total wall time: {(time.time()-overall_start)/60:.1f} min")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 7, 13, 42])
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    main(args.seeds, args.epochs)
