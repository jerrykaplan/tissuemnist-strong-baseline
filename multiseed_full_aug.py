"""
Multi-seed verification of ResNet-18 + full_aug on TissueMNIST.

Runs the full_aug config (rotation±15°, translate, flip, 40 epochs) across
3 additional seeds to check whether our 0.7335 result is robust or a
lucky single-seed.

Compares against the published SOTA (MedViTV2-large at 0.716, Applied
Soft Computing 2025) and the official ResNet-18 baseline at 0.676.
"""
import os, sys, time, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ResNet18_28, TissueDS

torch.set_num_threads(8)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NPZ_PATH    = '/workspace/ahd_tissue/tissuemnist.npz'
RESULTS_DIR = '/workspace/ahd_tissue/cnn_results'
ALL_CLASSES = list(range(8))
N_CLASSES = 8


def train_one_seed(seed, epochs=40, lr=1e-3, batch_size=128):
    """Run a single full_aug training with given seed."""
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

    history = []
    best_val_acc = 0.0
    best_test_at_best_val = 0.0
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

        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        test_correct = 0; test_total = 0
        per_class_correct = np.zeros(8, dtype=int)
        per_class_total   = np.zeros(8, dtype=int)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)
                for c in ALL_CLASSES:
                    m = (y == c)
                    per_class_total[c] += m.sum().item()
                    per_class_correct[c] += ((pred == y) & m).sum().item()
        test_acc = test_correct / test_total
        elapsed = time.time() - t0

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_acc': val_acc, 'test_acc': test_acc,
                        'per_class_acc': (per_class_correct/per_class_total).tolist(),
                        'elapsed': elapsed})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_at_best_val = test_acc
        # Reduce output verbosity — only print every 5 epochs + first/last
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:>2d}/{epochs}: loss={train_loss:.4f}  "
                  f"val={val_acc:.4f}  test={test_acc:.4f}  ({elapsed:.1f}s)")

    final = history[-1]
    print(f"  Final test: {final['test_acc']:.4f}, "
          f"best val: {best_val_acc:.4f} (test at best val: {best_test_at_best_val:.4f})")
    print(f"  Per-class: " + ' '.join(f'{v:.3f}' for v in final['per_class_acc']))
    return {'seed': seed, 'final_test_acc': final['test_acc'],
            'best_val_acc': best_val_acc,
            'best_test_at_best_val': best_test_at_best_val,
            'per_class_acc': final['per_class_acc'],
            'history': history}


def main(seeds, epochs):
    results = []
    for seed in seeds:
        t0 = time.time()
        result = train_one_seed(seed, epochs=epochs)
        result['wall_time'] = time.time() - t0
        results.append(result)
        print(f"\n  Wall time for seed {seed}: {result['wall_time']/60:.1f} min")

    # Save all results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'multiseed_full_aug.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    print(f"\nSaved to {out_path}")

    # Summary
    final_accs = np.array([r['final_test_acc'] for r in results])
    best_val_test_accs = np.array([r['best_test_at_best_val'] for r in results])

    print(f"\n{'='*60}")
    print(f"MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    print(f"  Seeds run this session: {[r['seed'] for r in results]}")
    print(f"")
    print(f"  Final test accuracies: " +
          ' '.join(f"{a:.4f}" for a in final_accs))
    print(f"  Mean: {final_accs.mean():.4f}, "
          f"Std: {final_accs.std():.4f}, "
          f"Range: [{final_accs.min():.4f}, {final_accs.max():.4f}]")
    print(f"")
    print(f"  Best-val test accuracies: " +
          ' '.join(f"{a:.4f}" for a in best_val_test_accs))
    print(f"  Mean: {best_val_test_accs.mean():.4f}, "
          f"Std: {best_val_test_accs.std():.4f}")

    print(f"\n{'='*60}")
    print(f"COMPARISON TO PUBLISHED BENCHMARKS")
    print(f"{'='*60}")
    print(f"  Official ResNet-18 (MedMNIST v2, Yang et al. 2023):  0.676")
    print(f"  AutoKeras (MedMNIST v2 best baseline, 2023):         0.703")
    print(f"  MedViTV2-large (Applied Soft Computing 2025 SOTA):   0.716")
    print(f"")
    print(f"  Our previous full_aug (seed 42):                     0.7335")
    print(f"  Our cascade_stage1 (seed 42, different pod):         0.7311")
    print(f"  This run mean ({len(seeds)} seeds):                     {final_accs.mean():.4f}")
    print(f"")
    print(f"  Δ vs published SOTA (0.716):  {final_accs.mean() - 0.716:+.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 7, 13],
                        help='Seeds to test (3 additional seeds vs original 42)')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    main(args.seeds, args.epochs)
