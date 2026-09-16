# Model Card: GaitJEPA (IJCB 2026)

## Model Details

- **Model Name**: GaitJEPA Encoder (DeepGaitV2JEPAEncoder)
- **Model Type**: Self-supervised spatiotemporal gait feature extractor
- **Paper**: *GaitJEPA: How far can we go with JEPA on binary silhouettes for gait recognition?* (IJCB 2026)
- **Authors**: Manuel J. Marin-Jimenez, Isabel Jimenez-Velasco, Rafael Munoz-Salinas
- **Release Version**: 1.0 (Minimal Inference Release)
- **Framework**: PyTorch (`torch >= 2.0.0`)
- **Repository**: https://github.com/AVAuco/gaitjepa
- **License**: MIT

### Architecture Summary

- **Backbone Mode**: Pseudo-3D (P3D) ResNet-like architecture (`BasicBlockP3D`)
- **Stage Layers**: `(1, 1, 1, 1)`
- **Channel Dimensions**: `(64, 128, 256, 512)`
- **Spatial Pooling**: Horizontal Pooling Pyramid (HPP) with 16 horizontal bins
- **Temporal Pooling**: Max-pooling across time
- **Projection Head**: Separate Linear Mappings per horizontal part bin (`SeparateFCs`, $16 \times 512 \to 256$)
- **Parameters**: ~8.17M parameters in the encoder

---

## Intended Use

- **Intended Domain**: Computer vision, gait analysis, biometric representation learning, health/clinical motion analysis.
- **Primary Use Cases**:
  - Extracting compact spatial and spatiotemporal representations from human silhouette sequences.
  - Initializing backbones for supervised or self-supervised downstream tasks (e.g. cross-view identification, covariate-robust recognition, medical gait anomaly detection).
- **Out of Scope / Misuse**:
  - The published checkpoint is not a ready-to-use identity verification or classification system; downstream linear probes or adaptation heads must be trained for specific tasks.
  - Surveillance applications without explicit user consent or in violation of biometric privacy regulations.

---

## Training Data & Regimen

- **Dataset**: Pretrained on a large-scale unlabeled silhouette corpus based on GaitLU-1M.
- **Objective**: Joint-Embedding Predictive Architecture (JEPA) combined with Latent Embedding World Model (LEWM) dynamics and SIGReg regularization.
- **Input Preprocessing**: Bounding box centering and normalized aspect ratio resized to $64 \times 44$ pixels.
- **Temporal Window**: Pretrained with 30-frame sequence windows.

---

## Technical Specifications & Input Contract

- **Input Tensor**: `[B, T, 1, 64, 44]` (Float32 in range $[0.0, 1.0]$).
- **Batch Size**: Arbitrary ($B \ge 1$).
- **Frame Count**: $T \ge 8$ recommended (native support for arbitrary length).
- **Outputs**:
  - `global_embedding`: `[B, 256]` (L2-normalized)
  - `pooled_part_embedding`: `[B, 256, 16]`
  - `sequence_part_tokens`: `[B, T, 16, 256]`

---


## Ethical & Privacy Considerations

Gait representations capture biometric characteristics. Deployments must:
- Comply with applicable biometric privacy laws (e.g., GDPR, CCPA).
- Ensure explicit subject consent and transparency.
- Be evaluated for fairness and demographic parity across age, sex, and clothing variations prior to any real-world use.
