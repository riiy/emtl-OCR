# Public emtl-OCR dataset

This directory contains the public, manually labeled four-digit synthetic CAPTCHA dataset used by the `emtl-OCR` research project.

## Contents

- `images/`: one copy of each JPEG image. Each filename is the complete SHA-256 digest of its unchanged image bytes, followed by `.jpg`.
- `manifest.json`: image IDs, canonical relative paths, fixed split membership, provenance metadata, and baseline membership constraints.
- `labels.csv`: the sole authoritative label source. It has exactly two columns: `image_id,label`.

## Dataset splits

| Split | Images | Use |
|---|---:|---|
| `train` | 880 | Model training |
| `validation` | 220 | Model selection and tuning |
| `test` | 200 | Public fixed final evaluation |
| **Total** | **1,300** | |

The 1,100 original labeled images are split into 880 training and 220 validation images using the historical source-stratified 80/20 procedure and seed `20261127`. The source categories and sample IDs are intentionally not published. Since those source categories are withheld, the seed and method alone cannot recreate the split. The explicit image membership in `manifest.json` is authoritative.

The 200 manually labeled test images retain their own fixed membership. Their labels are public, so this is a **public fixed test set**, not a confidential or secret blind test set. It must not be used for training, model selection, or hyperparameter tuning. Freeze a model before running the final evaluation, and report results honestly.

The original 1,387-image archive contained 287 images outside the original 1,100-row label CSV. Exactly 200 of those were manually labeled for the fixed test set. The other 87 candidate images (including five skipped images) are not included in this public dataset. The 20 remote-service exploration images are also excluded.

## Labels

`labels.csv` is UTF-8 CSV with exactly these columns:

```csv
image_id,label
<64-character-lowercase-sha256>,0042
```

Labels are four-character ASCII digit strings. Keep them as strings; leading zeroes are significant. Do not convert labels to integers.

`image_id` is the full SHA-256 digest of the corresponding image file. The image path is always `images/<image_id>.jpg`. The file bytes were copied from the original JPEG payloads without decoding and re-encoding; only the filename extension changed from `.img` to `.jpg`.

## Audit

From the repository root:

```bash
python tools/audit_dataset.py \
  --dataset datasets \
  --verify-images \
  --verify-labels \
  --verify-splits
```

The audit verifies the image inventory, image SHA-256 values, JPEG format and dimensions, manifest/label correspondence, four-digit labels, split counts, baseline membership, and the invariant that new images may only be added to training or validation. Existing baseline images cannot be moved or removed, and the public test membership cannot change.

## Build provenance

The maintainer build tool consumes the private source archive and frozen annotation artifacts. Those raw source files and annotation-operation logs are intentionally not distributed. The public dataset can be independently audited using the files in this directory.

Source archive SHA-256:

```text
90cb341f02c7db2658d5a25f3a21da431eb7108b3e56fb24da731ac06cb97494
```

The source image generator is no longer available. Historical generation used an operating-system-provided or open-source font, but the exact font family, version, and license cannot be recovered. This uncertainty is disclosed; it is not a claim that third-party rights have been independently cleared.

## License and attribution

The project applies **Creative Commons Attribution 4.0 International (CC BY 4.0)** to the dataset and labels to the extent of rights the project is entitled to license. See [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This notice does not grant rights in third-party fonts or other third-party material. The exact font provenance is unknown, and the project has not completed independent legal clearance of those rights. Reusers are responsible for assessing any third-party rights that may apply. The license statement is not legal advice or a warranty that no third-party restrictions exist.

When using the dataset, attribute it to **riiy/emtl-OCR** and include the CC BY 4.0 license link.
