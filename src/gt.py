import numpy as np

mask_path = "/Users/farah/datasets/brats/train/masks/mask_0.npy"
m = np.load(mask_path)

print("mask shape:", m.shape, "dtype:", m.dtype)
print("min/max:", m.min(), m.max())
print("unique overall:", np.unique(m)[:10], " ... total:", len(np.unique(m)))

# check per-channel occupancy
for c in range(m.shape[-1]):
    frac = (m[..., c] > 0.5).mean()
    print(f"channel {c} fraction ones: {frac:.4f}")

# check union occupancy (this is what your loader uses)
union = (m > 0.5).any(axis=-1)
print("union fraction ones:", union.mean())