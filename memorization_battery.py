"""
Memorization-prevention battery for TissueMNIST.

Hypothesis: augmentation's benefit on TissueMNIST is regularization
(preventing memorization), not invariance teaching. If true, simpler
non-geometric perturbations should also reach ~0.73.

Configurations tested (each with NO geometric augmentation):
  - control_noaug: baseline, no augmentation (expected ~0.668)
  - noise_001:     Gaussian noise std=0.01 (in pixel-normalized units)
  - noise_005:     Gaussian noise std=0.05
  - noise_010:     Gaussian noise std=0.10
  - noise_020:     Gaussian noise std=0.20
  - cutout_8:      Random 8x8 zero patch
  - mixup_02:      MixUp with alpha=0.2
  - labelsmooth:   Label smoothing 0.1 (no input perturbation)
  - wd_high:       Weight decay 0.01 (no input perturbation)

Reference: full_aug (rotation+translate+flip) reaches 0.7335 (mean
0.7342 across seeds). no_aug reaches 0.668. Gap to close: ~6.5 pp.

All runs use seed=42, 40 epochs, ResNet-18.
"""
import os, sys, time, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ResNet18_28, TissueDS

torch.set_num_threads(8)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NPZ_PATH    = '/workspace/ahd_tissue/tissuemnist.npz'
RESULTS_DIR = '/workspace/ahd_tissue/cnn_results'
N_CLASSES = 8


class TissueDS_perturbed(Dataset):
    """Dataset with configurable perturbation (no geometric augmentation)."""
    def __init__(self, X, y, perturbation=None, perturbation_params=None,
                 training=True):
        self.X = X  # uint8 28x28
        self.y = y.flatten().astype(np.int64)
        self.perturbation = perturbation
        self.params = perturbation_params or {}
        self.training = training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].astype(np.float32) / 255.0  # [0, 1]
        label = self.y[idx]

        if self.training and self.perturbation is not None:
            if self.perturbation == 'gaussian_noise':
                std = self.params.get('std', 0.05)
                noise = np.random.randn(*img.shape).astype(np.float32) * std
                img = img + noise
                img = np.clip(img, 0.0, 1.0)
            elif self.perturbation == 'cutout':
                patch_size = self.params.get('patch_size', 8)
                h, w = img.shape
                cy = np.random.randint(0, h)
                cx = np.random.randint(0, w)
                y0 = max(0, cy - patch_size // 2)
                y1 = min(h, cy + patch_size // 2)
                x0 = max(0, cx - patch_size // 2)
                x1 = min(w, cx + patch_size // 2)
                img = img.copy()
                img[y0:y1, x0:x1] = 0.0

        # Normalize to [-1, 1] for ResNet stem (matches original TissueDS)
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0), torch.tensor(label)


def mixup_batch(x, y, alpha=0.2, n_classes=8):
    """Apply MixUp to a batch. Returns mixed inputs and pair of (y1, y2, lam)."""
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(batch_size, device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    return x_mixed, y, y[perm], lam


def train_one_config(config_name, config, seed=42, epochs=40,
                      lr=1e-3, batch_size=128):
    """Train ResNet-18 with a given memorization-prevention config."""
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  Description: {config.get('description', 'N/A')}")
    print(f"{'='*60}")

    npz = np.load(NPZ_PATH)
    train_X = npz['train_X']; train_y = npz['train_y']
    val_X = npz['val_X']; val_y = npz['val_y']
    test_X = npz['test_X']; test_y = npz['test_y']

    train_ds = TissueDS_perturbed(
        train_X, train_y,
        perturbation=config.get('perturbation'),
        perturbation_params=config.get('perturbation_params'),
        training=True)
    val_ds   = TissueDS_perturbed(val_X,   val_y,   training=False)
    test_ds  = TissueDS_perturbed(test_X,  test_y,  training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2)

    model = ResNet18_28(num_classes=N_CLASSES, in_channels=1).to(DEVICE)

    wd = config.get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs // 2, 3 * epochs // 4], gamma=0.1)

    label_smoothing = config.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_mixup = config.get('use_mixup', False)
    mixup_alpha = config.get('mixup_alpha', 0.2)

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
            if use_mixup:
                x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
                out = model(x_mixed)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
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
                for c in range(N_CLASSES):
                    m = (y == c)
                    per_class_total[c] += m.sum().item()
                    per_class_correct[c] += ((pred == y) & m).sum().item()
        test_acc = test_correct / test_total
        elapsed = time.time() - t0

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_acc': val_acc, 'test_acc': test_acc,
                        'per_class_acc': (per_class_correct/per_class_total).tolist()})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_at_best_val = test_acc
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:>2d}/{epochs}: loss={train_loss:.4f}  "
                  f"val={val_acc:.4f}  test={test_acc:.4f}  ({elapsed:.1f}s)")

    final_test = history[-1]['test_acc']
    print(f"  Final test: {final_test:.4f}, best val: {best_val_acc:.4f} "
          f"(test at best val: {best_test_at_best_val:.4f})")
    print(f"  Per-class: " +
          ' '.join(f'{v:.3f}' for v in history[-1]['per_class_acc']))
    return {'config_name': config_name, 'config': config,
            'final_test_acc': final_test,
            'best_val_acc': best_val_acc,
            'best_test_at_best_val': best_test_at_best_val,
            'per_class_acc': history[-1]['per_class_acc'],
            'history': history}


CONFIGS = [
    ('control_noaug', {
        'description': 'No augmentation, no perturbation (control)',
    }),
    ('noise_001', {
        'description': 'Gaussian noise std=0.01',
        'perturbation': 'gaussian_noise',
        'perturbation_params': {'std': 0.01},
    }),
    ('noise_005', {
        'description': 'Gaussian noise std=0.05',
        'perturbation': 'gaussian_noise',
        'perturbation_params': {'std': 0.05},
    }),
    ('noise_010', {
        'description': 'Gaussian noise std=0.10',
        'perturbation': 'gaussian_noise',
        'perturbation_params': {'std': 0.10},
    }),
    ('noise_020', {
        'description': 'Gaussian noise std=0.20',
        'perturbation': 'gaussian_noise',
        'perturbation_params': {'std': 0.20},
    }),
    ('cutout_8', {
        'description': 'Cutout 8x8 zero patch',
        'perturbation': 'cutout',
        'perturbation_params': {'patch_size': 8},
    }),
    ('mixup_02', {
        'description': 'MixUp alpha=0.2 (no input perturbation)',
        'use_mixup': True,
        'mixup_alpha': 0.2,
    }),
    ('labelsmooth_01', {
        'description': 'Label smoothing 0.1, no other perturbation',
        'label_smoothing': 0.1,
    }),
    ('wd_high', {
        'description': 'Weight decay 0.01, no other perturbation',
        'weight_decay': 0.01,
    }),
]


def main(epochs):
    results = []
    overall_start = time.time()
    for config_name, config in CONFIGS:
        t0 = time.time()
        result = train_one_config(config_name, config, epochs=epochs)
        result['wall_time'] = time.time() - t0
        results.append(result)
        print(f"  Wall time: {result['wall_time']/60:.1f} min")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'memorization_battery.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    print(f"\nSaved to {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"MEMORIZATION-PREVENTION BATTERY SUMMARY")
    print(f"{'='*70}")
    print(f"  Reference: no_aug baseline (control) ~ 0.668")
    print(f"  Reference: full_aug (rotation+translate+flip) ~ 0.7335")
    print(f"  Gap to close: ~6.5 pp")
    print(f"")
    print(f"  {'Config':<22} {'Final test':>11} {'Best val→test':>15} "
          f"{'Δ vs noaug':>12} {'Δ vs full':>10}")
    print(f"  {'-'*22} {'-'*11} {'-'*15} {'-'*12} {'-'*10}")
    NOAUG = 0.668
    FULLAUG = 0.7335
    for r in results:
        acc = r['final_test_acc']
        bv = r['best_test_at_best_val']
        d_noaug = acc - NOAUG
        d_full = acc - FULLAUG
        print(f"  {r['config_name']:<22} {acc:>11.4f} {bv:>15.4f} "
              f"{d_noaug:>+12.4f} {d_full:>+10.4f}")
    print(f"")
    print(f"  Total wall time: {(time.time()-overall_start)/60:.1f} min")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    main(args.epochs)
