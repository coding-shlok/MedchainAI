"""Dataset loading, tiered by cost.

  synthetic  Gaussian blobs, no download. CI and protocol debugging.
  medmnist   28x28 or 64x64 medical images, seconds to download. Dev tier -- use
             it to shake out every experiment before spending GPU time.
  isic2019   PRIMARY for the paper. 25,331 dermoscopic images, 8 classes, severe
             class imbalance (melanoma is a small minority), ~9 GB.
  chestxray  NIH ChestX-ray14, optional SECOND dataset for generalisation claims.
             112k images, 14 labels, MULTI-LABEL -- needs BCEWithLogitsLoss and
             AUC, not accuracy. Do not report accuracy on it; with sparse
             positives, predicting all-negative scores above 90%.

ISIC IS THE PRIMARY FOR A REASON: real class imbalance means macro-F1 and
per-class recall are meaningful, so a poisoning attack that suppresses the
melanoma class produces a clinically legible harm rather than an abstract
accuracy drop. That is a much stronger motivating story than "accuracy fell 3%".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    name: str
    image_shape: Optional[Tuple[int, int, int]] = None

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.X_train.shape[1:]))

    def class_balance(self) -> dict:
        counts = np.bincount(self.y_train, minlength=self.num_classes)
        return {"counts": counts.tolist(),
                "imbalance_ratio": float(counts.max() / max(1, counts.min()))}


def load_synthetic(num_classes: int = 4, dim: int = 30, n: int = 3000,
                   separation: float = 1.1, noise: float = 1.6,
                   seed: int = 0) -> DatasetBundle:
    """Deliberately overlapping classes. Well-separated blobs hit ~100% accuracy
    and then no attack or defence shows any measurable effect, which makes the
    smoke test silently useless."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, separation, (num_classes, dim))
    y = rng.integers(0, num_classes, n)
    X = (centers[y] + rng.normal(0, noise, (n, dim))).astype(np.float32)
    split = int(0.8 * n)
    return DatasetBundle(X[:split], y[:split], X[split:], y[split:],
                         num_classes, "synthetic")


def load_medmnist(flag: str = "pathmnist", size: int = 28, root: str = "data",
                  flatten: bool = True) -> DatasetBundle:
    """Requires `pip install medmnist`. Downloads on first use."""
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as e:
        raise ImportError("pip install medmnist") from e

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    Path(root).mkdir(parents=True, exist_ok=True)
    tr = DataClass(split="train", download=True, root=root, size=size)
    te = DataClass(split="test", download=True, root=root, size=size)

    def prep(ds):
        X = ds.imgs.astype(np.float32) / 255.0
        if X.ndim == 3:
            X = X[..., None]
        if X.shape[-1] == 1:
            X = np.repeat(X, 3, axis=-1)
        y = ds.labels.ravel().astype(int)
        return (X.reshape(X.shape[0], -1) if flatten else X), y

    Xtr, ytr = prep(tr)
    Xte, yte = prep(te)
    return DatasetBundle(Xtr, ytr, Xte, yte, len(info["label"]), f"medmnist:{flag}",
                         image_shape=(size, size, 3))


class _ImagePathDataset:
    """Map-style dataset over (path, label) pairs, shared by every torch-tier
    image loader below. MUST live at module scope, not as a class nested
    inside a loader function: DataLoader(num_workers>0) pickles the Dataset to
    hand it to worker subprocesses, and a nested class is a closure that
    cannot be pickled -- it fails with `Can't pickle local object` the moment
    you set num_workers >= 1, while silently working at num_workers=0. Torch
    and PIL are imported lazily inside the methods so importing this module
    stays torch-free until you actually iterate one of these."""

    def __init__(self, paths, labels, tf):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), torch.tensor(int(self.labels[i]))


MALIGNANT_CLASSES = {"MEL", "BCC", "SCC"}
"""Melanoma, basal cell carcinoma, squamous cell carcinoma -- the three ISIC 2019
categories that are actual cancers. The standard binary framing used across the
skin-lesion literature. AK (actinic keratosis) is deliberately left OUT: it is
precancerous, not cancer, and the common convention groups it with benign. That
is a real judgment call, not a fact -- say so if asked, and if a paper you're
citing draws the line differently, match theirs instead of this one."""


def isic2019_torch_datasets(root: str = "data/isic2019", image_size: int = 224,
                            val_fraction: float = 0.2, seed: int = 0,
                            max_samples: Optional[int] = None, binary: bool = False):
    """Torch Datasets for the primary experiments.

    The CSV is always required:
        <root>/ISIC_2019_Training_GroundTruth.csv

    Images can be laid out either way -- both are real ISIC 2019 repackagings
    people actually download:
      flat:           <root>/ISIC_2019_Training_Input/<image>.jpg
      class-foldered: <root>/<CLASS>/<image>.jpg   (CLASS in MEL, NV, BCC, AK,
                       BKL, DF, VASC, SCC -- the CSV's own column names)
    Detected automatically; you don't need to reorganise whichever one you have.

    Download: https://challenge.isic-archive.com/data/#2019

    THE SPLIT IS PATIENT-AGNOSTIC AND THAT IS A REAL LIMITATION. ISIC 2019 ships
    no reliable patient identifier, so multiple lesions from one patient can land
    on both sides of the split and inflate test accuracy. Say so in the paper.
    If you later use ISIC 2020, which does carry patient_id, group-split on it.

    `max_samples` subsamples (stratified-ish via the same shuffle used for the
    split) BEFORE the train/val split -- use it for a fast smoke test of the
    pipeline (minutes, not hours) before committing to a full run. Leave it
    None for the real paper numbers.

    `binary=True` collapses the 8 diagnostic classes to malignant/benign (see
    MALIGNANT_CLASSES) -- a coarser, different task from the multiclass one,
    not a variant of the same result. Report them as separate numbers, never
    as "improved accuracy" on the same benchmark.
    """
    import pandas as pd
    from torchvision import transforms

    root = Path(root)
    csv = root / "ISIC_2019_Training_GroundTruth.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Download ISIC 2019 from "
            f"https://challenge.isic-archive.com/data/#2019 and unpack into {root}")

    df = pd.read_csv(csv)
    # UNK is part of the official CSV schema but is always 0 for training images
    # (it's reserved for test-set predictions) -- keeping it as a 9th, permanently
    # empty class wastes an output neuron and breaks balanced-loss weighting
    # (divide-by-a-clipped-zero). Drop any label column with zero positives so
    # this works whether or not a given release includes it.
    label_cols = [c for c in df.columns if c != "image" and df[c].sum() > 0]
    labels = df[label_cols].values.argmax(axis=1)
    num_classes = len(label_cols)

    if binary:
        # Map by CLASS NAME, not by the integer position argmax just produced --
        # label_cols is alphabetically sorted (AK, BCC, BKL, ...), so index
        # order does not match any meaningful malignant/benign grouping.
        is_malignant = np.array([label_cols[i] in MALIGNANT_CLASSES for i in labels])
        labels = is_malignant.astype(int)  # 0=benign, 1=malignant
        num_classes = 2

    img_dir = root / "ISIC_2019_Training_Input"
    if img_dir.exists():
        paths = [str(img_dir / f"{n}.jpg") for n in df["image"]]
    else:
        # Class-foldered layout: index every image once, then look each one up
        # by id. Cheap even at 25k files (one os.walk-equivalent pass).
        index: Dict[str, str] = {}
        for cls_dir in root.iterdir():
            if cls_dir.is_dir():
                for p in cls_dir.glob("*.jpg"):
                    index[p.stem] = str(p)
        missing = [n for n in df["image"] if n not in index]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images from the CSV were not found under {root} "
                f"(neither {img_dir} nor any <root>/<CLASS>/ subfolder). First "
                f"missing: {missing[:3]}. Check the download unpacked correctly.")
        paths = [index[n] for n in df["image"]]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    if max_samples is not None:
        idx = idx[:max_samples]
    n_val = int(val_fraction * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # ImageNet statistics, because the backbones are ImageNet-pretrained.
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size),
        transforms.ToTensor(), norm])

    train_paths = [paths[j] for j in train_idx]
    val_paths = [paths[j] for j in val_idx]
    return (_ImagePathDataset(train_paths, labels[train_idx], train_tf),
            _ImagePathDataset(val_paths, labels[val_idx], eval_tf),
            labels[train_idx], num_classes)


def isic_folder_torch_datasets(root: str = "Dataset", image_size: int = 224, seed: int = 0,
                               max_samples: Optional[int] = None):
    """Torch Datasets for a pre-split, folder-per-class ISIC derivative:

        <root>/Train/<class_name>/*.jpg
        <root>/Test/<class_name>/*.jpg

    This is the common Kaggle repackaging of a subset of ISIC images (~2.4k
    images, 9 classes, Train/Test already split) -- NOT the official ISIC 2019
    CSV release (25k images, 8 classes) that `isic2019_torch_datasets` above
    targets. The class semantics differ: this package splits benign keratosis
    into "seborrheic keratosis" and "pigmented benign keratosis" as separate
    folders, where the official ISIC 2019 taxonomy keeps them as one BKL class.
    Say so if you cite ISIC 2019 numbers from this dataset -- they are not the
    same benchmark.

    Unlike isic2019_torch_datasets, the train/val split already exists on disk
    and is respected as-is (not re-split), so results are comparable to anyone
    else using this same package. `max_samples` subsamples the TRAIN split
    only, for a fast smoke test.
    """
    from torchvision import transforms

    root = Path(root)
    train_dir, test_dir = root / "Train", root / "Test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"expected {train_dir} and {test_dir} (one subfolder per class in each)")

    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class subfolders found in {train_dir}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def scan(d: Path):
        paths, labels = [], []
        for c in classes:
            for p in sorted((d / c).glob("*.jpg")):
                paths.append(str(p))
                labels.append(class_to_idx[c])
        return paths, np.array(labels, dtype=int)

    train_paths, train_labels = scan(train_dir)
    test_paths, test_labels = scan(test_dir)

    if max_samples is not None and max_samples < len(train_paths):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(train_paths))[:max_samples]
        train_paths = [train_paths[i] for i in idx]
        train_labels = train_labels[idx]

    # ImageNet statistics, because the backbones are ImageNet-pretrained.
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size),
        transforms.ToTensor(), norm])

    train_ds = _ImagePathDataset(train_paths, train_labels, train_tf)
    test_ds = _ImagePathDataset(test_paths, test_labels, eval_tf)
    return train_ds, test_ds, train_labels, len(classes), classes


def load_dataset(name: str, **kwargs) -> DatasetBundle:
    if name == "synthetic":
        return load_synthetic(**kwargs)
    if name.startswith("medmnist"):
        flag = name.split(":", 1)[1] if ":" in name else "pathmnist"
        return load_medmnist(flag, **kwargs)
    raise ValueError(f"'{name}' is not a NumPy-tier dataset. ISIC and ChestX-ray "
                     f"are torch-tier: use isic2019_torch_datasets() with TorchClient.")
