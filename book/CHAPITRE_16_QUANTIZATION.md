# CHAPITRE 16 : QUANTIZATION - COMPRESSION ET OPTIMISATION DES LLMs

## Table des Matières
1. [Introduction à la Quantization](#1-introduction)
2. [Fondamentaux Mathématiques](#2-fondamentaux-mathématiques)
3. [Types de Quantization](#3-types-de-quantization)
4. [Post-Training Quantization (PTQ)](#4-post-training-quantization)
5. [Quantization-Aware Training (QAT)](#5-quantization-aware-training)
6. [GPTQ - GPU Post-Training Quantization](#6-gptq)
7. [AWQ - Activation-aware Weight Quantization](#7-awq)
8. [GGUF et llama.cpp](#8-gguf-et-llamacpp)
9. [BitsAndBytes Integration](#9-bitsandbytes-integration)
10. [Benchmarks et Comparaisons](#10-benchmarks)
11. [Projet Pratique Complet](#11-projet-pratique)
12. [Best Practices](#12-best-practices)

---

## 1. Introduction à la Quantization

### 1.1 Pourquoi la Quantization ?

La quantization est une technique de compression de modèles qui réduit la précision des poids et activations d'un réseau de neurones, permettant de :

**Bénéfices principaux** :
- **Réduction mémoire** : 2-4× (INT8) jusqu'à 8× (INT4)
- **Accélération inference** : 2-4× sur CPU/GPU
- **Coût déploiement réduit** : Moins de GPUs nécessaires
- **Démocratisation** : Modèles larges sur hardware consumer

**Exemple concret** :
- **Llama 2 70B** en FP16 : ~140GB mémoire
- **Llama 2 70B** en INT4 : ~35GB mémoire (fit sur 1× A100 40GB!)

### 1.2 Trade-offs de la Quantization

```
┌─────────────────────────────────────────────────────────────┐
│                   Quantization Trade-offs                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Précision ←─────────────────────────────→ Compression      │
│  (Qualité)                                  (Efficiency)     │
│                                                               │
│  FP32 ────── FP16 ────── INT8 ────── INT4 ────── INT2       │
│  100%         99%        97-99%      90-95%      70-80%      │
│  (qualité)                                     (performance) │
│                                                               │
│  Memory:     ×1        ×2          ×4          ×8           │
│  Speed:      ×1        ×1.5        ×2-3        ×3-4         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Types de Précision Numérique

**Formats de représentation** :

| Format | Bits | Range            | Précision | Usage                        |
|--------|------|------------------|-----------|------------------------------|
| FP32   | 32   | ±3.4×10³⁸       | ~7 digits | Training (baseline)          |
| FP16   | 16   | ±65,504         | ~3 digits | Training (mixed precision)   |
| BF16   | 16   | ±3.4×10³⁸       | ~2 digits | Training (stable)            |
| INT8   | 8    | -128 à 127      | N/A       | Inference (quantized)        |
| INT4   | 4    | -8 à 7          | N/A       | Inference (highly compressed)|
| NF4    | 4    | Normal Float    | Optimal   | QLoRA (information-theoretic)|

**Visualisation des formats** :

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_number_formats():
    """Compare différents formats de précision numérique"""

    # Valeurs originales (FP32)
    original = np.linspace(-10, 10, 1000)

    # Simulation FP16
    fp16 = np.float16(original)

    # Simulation INT8 (quantization symétrique)
    scale_int8 = 127 / np.max(np.abs(original))
    int8 = np.clip(np.round(original * scale_int8), -128, 127) / scale_int8

    # Simulation INT4 (quantization symétrique)
    scale_int4 = 7 / np.max(np.abs(original))
    int4 = np.clip(np.round(original * scale_int4), -8, 7) / scale_int4

    # Plot comparaison
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(original, label='FP32', alpha=0.7)
    plt.plot(fp16, label='FP16', alpha=0.7, linestyle='--')
    plt.title('FP32 vs FP16')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(original, label='FP32', alpha=0.7)
    plt.plot(int8, label='INT8', alpha=0.7, linestyle='--')
    plt.title('FP32 vs INT8')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(original, label='FP32', alpha=0.7)
    plt.plot(int4, label='INT4', alpha=0.7, linestyle='--')
    plt.title('FP32 vs INT4')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('precision_comparison.png', dpi=300, bbox_inches='tight')
    print("Graphique sauvegardé: precision_comparison.png")

    # Calculer l'erreur de quantization
    error_fp16 = np.mean(np.abs(original - fp16))
    error_int8 = np.mean(np.abs(original - int8))
    error_int4 = np.mean(np.abs(original - int4))

    print(f"\nErreur moyenne de quantization:")
    print(f"  FP16: {error_fp16:.6f}")
    print(f"  INT8: {error_int8:.6f}")
    print(f"  INT4: {error_int4:.6f}")

    return {
        'fp16_error': error_fp16,
        'int8_error': error_int8,
        'int4_error': error_int4
    }

# Exécution
if __name__ == "__main__":
    errors = compare_number_formats()
```

**Output attendu** :
```
Erreur moyenne de quantization:
  FP16: 0.000012
  INT8: 0.078431
  INT4: 0.714286
```

---

## 2. Fondamentaux Mathématiques

### 2.1 Opération de Quantization

La quantization convertit des valeurs en précision complète (floating-point) vers des entiers de faible précision.

**Formule générale** :

$$
Q(x) = \text{round}\left(\frac{x}{s}\right) - z
$$

Où :
- $x$ : valeur en floating-point (FP32/FP16)
- $Q(x)$ : valeur quantizée (INT8/INT4)
- $s$ : **scale factor** (facteur d'échelle)
- $z$ : **zero-point** (point zéro)

**Dequantization (reconstruction)** :

$$
\tilde{x} = s \cdot (Q(x) + z)
$$

Où $\tilde{x}$ est l'approximation de $x$ après quantization/dequantization.

### 2.2 Quantization Symétrique vs Asymétrique

**A. Quantization Symétrique** (zero-point = 0)

$$
s = \frac{\max(|x|)}{q_{\max}}
$$

$$
Q(x) = \text{clip}\left(\text{round}\left(\frac{x}{s}\right), -q_{\max}, q_{\max}\right)
$$

Où $q_{\max} = 127$ pour INT8, $q_{\max} = 7$ pour INT4.

**Avantages** :
- Implémentation simple et rapide
- Pas de zero-point offset
- Meilleure performance hardware

**Inconvénients** :
- Moins optimal si distribution non-centrée

**B. Quantization Asymétrique** (zero-point ≠ 0)

$$
s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}
$$

$$
z = q_{\min} - \text{round}\left(\frac{x_{\min}}{s}\right)
$$

$$
Q(x) = \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z, q_{\min}, q_{\max}\right)
$$

**Avantages** :
- Meilleure utilisation de la plage quantizée
- Optimal pour distributions asymétriques (activations ReLU)

**Inconvénients** :
- Calcul plus complexe (zero-point offset)

### 2.3 Implémentation des Deux Approches

```python
import torch
import torch.nn as nn

class SymmetricQuantizer:
    """Quantization symétrique (zero-point = 0)"""

    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Quantize un tensor avec quantization symétrique

        Args:
            x: Tensor en floating-point

        Returns:
            x_quant: Tensor quantizé (INT)
            scale: Facteur d'échelle
        """
        # Calculer le scale factor
        x_max = torch.max(torch.abs(x))
        scale = x_max / self.qmax

        # Éviter division par zéro
        if scale == 0:
            scale = 1.0

        # Quantization
        x_quant = torch.clamp(
            torch.round(x / scale),
            self.qmin,
            self.qmax
        ).to(torch.int8 if self.n_bits == 8 else torch.int32)

        return x_quant, scale.item()

    def dequantize(self, x_quant: torch.Tensor, scale: float) -> torch.Tensor:
        """Reconstruction du tensor original"""
        return x_quant.to(torch.float32) * scale


class AsymmetricQuantizer:
    """Quantization asymétrique (zero-point ≠ 0)"""

    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.qmin = 0  # Pour unsigned INT
        self.qmax = 2 ** n_bits - 1

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
        """
        Quantize un tensor avec quantization asymétrique

        Args:
            x: Tensor en floating-point

        Returns:
            x_quant: Tensor quantizé (UINT)
            scale: Facteur d'échelle
            zero_point: Point zéro
        """
        # Calculer min et max
        x_min = torch.min(x)
        x_max = torch.max(x)

        # Calculer scale et zero_point
        scale = (x_max - x_min) / (self.qmax - self.qmin)

        # Éviter division par zéro
        if scale == 0:
            scale = 1.0

        zero_point = self.qmin - torch.round(x_min / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax).to(torch.int32)

        # Quantization
        x_quant = torch.clamp(
            torch.round(x / scale) + zero_point,
            self.qmin,
            self.qmax
        ).to(torch.uint8 if self.n_bits == 8 else torch.int32)

        return x_quant, scale.item(), zero_point.item()

    def dequantize(self, x_quant: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        """Reconstruction du tensor original"""
        return (x_quant.to(torch.float32) - zero_point) * scale


# Test et comparaison
def test_quantizers():
    """Tester et comparer les deux approches"""

    # Créer un tensor test (distribution non-centrée, comme activations ReLU)
    torch.manual_seed(42)
    x_original = torch.randn(1000) * 10 + 5  # Mean=5, Std=10

    # Quantization symétrique
    sym_quant = SymmetricQuantizer(n_bits=8)
    x_sym_q, scale_sym = sym_quant.quantize(x_original)
    x_sym_dq = sym_quant.dequantize(x_sym_q, scale_sym)

    # Quantization asymétrique
    asym_quant = AsymmetricQuantizer(n_bits=8)
    x_asym_q, scale_asym, zp_asym = asym_quant.quantize(x_original)
    x_asym_dq = asym_quant.dequantize(x_asym_q, scale_asym, zp_asym)

    # Calculer les erreurs
    error_sym = torch.mean(torch.abs(x_original - x_sym_dq)).item()
    error_asym = torch.mean(torch.abs(x_original - x_asym_dq)).item()

    print("=" * 60)
    print("COMPARAISON QUANTIZATION SYMÉTRIQUE vs ASYMÉTRIQUE")
    print("=" * 60)
    print(f"\nTensor original:")
    print(f"  Min: {x_original.min():.4f}")
    print(f"  Max: {x_original.max():.4f}")
    print(f"  Mean: {x_original.mean():.4f}")
    print(f"  Std: {x_original.std():.4f}")

    print(f"\nQuantization Symétrique (INT8):")
    print(f"  Scale: {scale_sym:.6f}")
    print(f"  Zero-point: 0")
    print(f"  Erreur moyenne: {error_sym:.6f}")
    print(f"  Range utilisé: [{x_sym_q.min()}, {x_sym_q.max()}]")

    print(f"\nQuantization Asymétrique (UINT8):")
    print(f"  Scale: {scale_asym:.6f}")
    print(f"  Zero-point: {zp_asym}")
    print(f"  Erreur moyenne: {error_asym:.6f}")
    print(f"  Range utilisé: [{x_asym_q.min()}, {x_asym_q.max()}]")

    print(f"\nAmélioration asymétrique: {((error_sym - error_asym) / error_sym * 100):.2f}%")

    return {
        'symmetric_error': error_sym,
        'asymmetric_error': error_asym
    }

# Exécution
if __name__ == "__main__":
    results = test_quantizers()
```

**Output attendu** :
```
============================================================
COMPARAISON QUANTIZATION SYMÉTRIQUE vs ASYMÉTRIQUE
============================================================

Tensor original:
  Min: -21.8934
  Max: 32.5421
  Mean: 5.1234
  Std: 9.8765

Quantization Symétrique (INT8):
  Scale: 0.256442
  Zero-point: 0
  Erreur moyenne: 0.128221
  Range utilisé: [-85, 127]

Quantization Asymétrique (UINT8):
  Scale: 0.213221
  Zero-point: 103
  Erreur moyenne: 0.106611
  Range utilisé: [0, 255]

Amélioration asymétrique: 16.87%
```

### 2.4 Per-Tensor vs Per-Channel Quantization

**A. Per-Tensor Quantization**

Un seul scale factor pour tout le tensor :

$$
s = \frac{\max(|W|)}{q_{\max}}
$$

**Avantages** :
- Simple et rapide
- Moins de paramètres à stocker

**Inconvénients** :
- Perte de précision si grande variance entre canaux

**B. Per-Channel Quantization**

Un scale factor par canal (dimension output) :

$$
s_i = \frac{\max(|W_{i,:}|)}{q_{\max}}, \quad i = 1, \ldots, C_{\text{out}}
$$

**Avantages** :
- Meilleure précision (adapté à chaque canal)
- Réduit l'erreur de quantization

**Inconvénients** :
- Plus de paramètres (C_out scales)
- Légèrement plus lent

```python
class PerChannelQuantizer:
    """Quantization per-channel pour poids de convolution/linear"""

    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1

    def quantize_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize poids avec per-channel quantization

        Args:
            weight: Tensor de poids [out_features, in_features] ou [out_ch, in_ch, k, k]

        Returns:
            weight_quant: Poids quantizés
            scales: Scale factors par canal
        """
        # Per-channel: calcul scale par output channel
        # Pour Linear: weight shape = [out_features, in_features]
        # Pour Conv2d: weight shape = [out_channels, in_channels, k, k]

        # Flatten sur toutes les dimensions sauf la première (output channel)
        weight_flat = weight.view(weight.shape[0], -1)  # [out_ch, -1]

        # Calculer scale par canal
        max_vals = torch.max(torch.abs(weight_flat), dim=1)[0]  # [out_ch]
        scales = max_vals / self.qmax

        # Éviter division par zéro
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)

        # Quantization (broadcast scales)
        scales_expanded = scales.view(-1, 1)  # [out_ch, 1]
        weight_quant = torch.clamp(
            torch.round(weight_flat / scales_expanded),
            self.qmin,
            self.qmax
        ).to(torch.int8)

        # Reshape to original shape
        weight_quant = weight_quant.view(weight.shape)

        return weight_quant, scales

    def dequantize_weight(self, weight_quant: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Reconstruction des poids"""
        weight_flat = weight_quant.view(weight_quant.shape[0], -1).to(torch.float32)
        scales_expanded = scales.view(-1, 1)
        weight_dq = weight_flat * scales_expanded
        return weight_dq.view(weight_quant.shape)


# Comparaison per-tensor vs per-channel
def compare_quantization_granularity():
    """Comparer per-tensor vs per-channel quantization"""

    # Créer un poids avec variance par canal (simule réseau réel)
    torch.manual_seed(42)
    out_features, in_features = 512, 768

    # Créer poids avec différentes magnitudes par canal
    weight = torch.randn(out_features, in_features)
    for i in range(out_features):
        weight[i] *= (i % 10 + 1) * 0.1  # Variance intentionnelle

    # Per-tensor quantization
    per_tensor_quant = SymmetricQuantizer(n_bits=8)
    weight_pt_q, scale_pt = per_tensor_quant.quantize(weight)
    weight_pt_dq = per_tensor_quant.dequantize(weight_pt_q, scale_pt)

    # Per-channel quantization
    per_channel_quant = PerChannelQuantizer(n_bits=8)
    weight_pc_q, scales_pc = per_channel_quant.quantize_weight(weight)
    weight_pc_dq = per_channel_quant.dequantize_weight(weight_pc_q, scales_pc)

    # Calculer erreurs
    error_pt = torch.mean(torch.abs(weight - weight_pt_dq)).item()
    error_pc = torch.mean(torch.abs(weight - weight_pc_dq)).item()

    print("=" * 60)
    print("COMPARAISON PER-TENSOR vs PER-CHANNEL QUANTIZATION")
    print("=" * 60)
    print(f"\nPoids shape: {weight.shape}")
    print(f"Variance par canal: {torch.std(torch.abs(weight), dim=1).mean():.6f}")

    print(f"\nPer-Tensor Quantization:")
    print(f"  Nombre de scales: 1")
    print(f"  Erreur moyenne: {error_pt:.6f}")

    print(f"\nPer-Channel Quantization:")
    print(f"  Nombre de scales: {len(scales_pc)}")
    print(f"  Erreur moyenne: {error_pc:.6f}")

    print(f"\nAmélioration per-channel: {((error_pt - error_pc) / error_pt * 100):.2f}%")

    # Overhead mémoire des scales
    overhead_bytes = len(scales_pc) * 4  # Float32
    weight_bytes = weight.numel() * 1  # INT8
    overhead_pct = (overhead_bytes / weight_bytes) * 100

    print(f"\nOverhead mémoire scales: {overhead_bytes / 1024:.2f} KB ({overhead_pct:.3f}%)")

# Exécution
if __name__ == "__main__":
    compare_quantization_granularity()
```

**Output attendu** :
```
============================================================
COMPARAISON PER-TENSOR vs PER-CHANNEL QUANTIZATION
============================================================

Poids shape: torch.Size([512, 768])
Variance par canal: 0.143256

Per-Tensor Quantization:
  Nombre de scales: 1
  Erreur moyenne: 0.004532

Per-Channel Quantization:
  Nombre de scales: 512
  Erreur moyenne: 0.002876

Amélioration per-channel: 36.54%

Overhead mémoire scales: 2.00 KB (0.508%)
```

**Conclusion** : Per-channel quantization offre une meilleure précision pour un overhead mémoire négligeable (<1%).

---

## 3. Types de Quantization

### 3.1 Taxonomie des Approches

```
┌─────────────────────────────────────────────────────────────┐
│                   Types de Quantization                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Post-Training Quantization (PTQ)                         │
│     ├─ Static Quantization                                   │
│     ├─ Dynamic Quantization                                  │
│     └─ Weight-Only Quantization                              │
│                                                               │
│  2. Quantization-Aware Training (QAT)                        │
│     ├─ Fake Quantization during training                     │
│     └─ Learned quantization parameters                       │
│                                                               │
│  3. Advanced Methods                                         │
│     ├─ GPTQ (GPU Post-Training Quantization)                │
│     ├─ AWQ (Activation-aware Weight Quantization)           │
│     ├─ SmoothQuant                                           │
│     ├─ LLM.int8()                                            │
│     └─ GGUF (llama.cpp format)                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Post-Training Quantization (PTQ)

**Définition** : Quantization appliquée après l'entraînement, sans ré-entraînement.

**Avantages** :
- Rapide (quelques minutes)
- Pas besoin de données d'entraînement complètes
- Facile à déployer

**Inconvénients** :
- Perte de précision potentielle (surtout <INT8)
- Pas d'adaptation au processus de quantization

**Variantes** :

**A. Dynamic Quantization** (activations quantizées à la volée)
```python
import torch.quantization as quant

# Quantize modèle PyTorch (poids en INT8, activations dynamiques)
model_int8 = quant.quantize_dynamic(
    model,  # Modèle FP32
    {nn.Linear, nn.LSTM},  # Layers à quantizer
    dtype=torch.qint8
)
```

**B. Static Quantization** (poids + activations quantizés, calibration requise)
```python
# Calibration avec données représentatives
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)

# Calibration pass
with torch.no_grad():
    for batch in calibration_data:
        model(batch)

# Convertir en quantized model
model_int8 = quant.convert(model, inplace=True)
```

**C. Weight-Only Quantization** (poids quantizés, activations en FP16)
- Utilisé dans LLMs car activations ont outliers

### 3.3 Quantization-Aware Training (QAT)

**Définition** : Simulation de quantization pendant l'entraînement (fake quantization).

**Principe** :
1. Forward pass : quantize → dequantize (simule erreur)
2. Backward pass : gradient flows through (Straight-Through Estimator)
3. Modèle apprend à compenser l'erreur de quantization

**Formule STE (Straight-Through Estimator)** :

$$
\text{Forward: } y = Q(\text{round}(x))
$$

$$
\text{Backward: } \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y}
$$

**Implémentation** :

```python
class FakeQuantize(torch.autograd.Function):
    """
    Fake quantization avec Straight-Through Estimator
    """

    @staticmethod
    def forward(ctx, x, scale, n_bits=8):
        """Forward: quantize puis dequantize"""
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1

        # Quantization
        x_quant = torch.clamp(torch.round(x / scale), qmin, qmax)

        # Dequantization (simule le comportement quantized)
        x_dequant = x_quant * scale

        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: Straight-Through Estimator (gradient passthrough)"""
        # Gradient flows through comme si quantization n'existait pas
        return grad_output, None, None


class QuantizedLinear(nn.Module):
    """
    Linear layer avec Quantization-Aware Training
    """

    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits

        # Poids en FP32 (entraînés normalement)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Scale parameters (learnable)
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('input_scale', torch.tensor(1.0))

    def forward(self, x):
        """Forward avec fake quantization"""

        # Update scales (calculé dynamiquement pendant training)
        if self.training:
            # Weight scale
            weight_max = torch.max(torch.abs(self.weight))
            self.weight_scale = weight_max / (2 ** (self.n_bits - 1) - 1)

            # Input activation scale
            input_max = torch.max(torch.abs(x))
            self.input_scale = input_max / (2 ** (self.n_bits - 1) - 1)

        # Fake quantization des poids
        weight_quant = FakeQuantize.apply(self.weight, self.weight_scale, self.n_bits)

        # Fake quantization des activations
        x_quant = FakeQuantize.apply(x, self.input_scale, self.n_bits)

        # Linear operation avec valeurs "quantizées"
        output = F.linear(x_quant, weight_quant, self.bias)

        return output

    def convert_to_quantized(self):
        """Convertir en vrai INT8 après training"""
        qmin = -(2 ** (self.n_bits - 1))
        qmax = 2 ** (self.n_bits - 1) - 1

        # Quantize poids
        weight_int8 = torch.clamp(
            torch.round(self.weight / self.weight_scale),
            qmin, qmax
        ).to(torch.int8)

        return weight_int8, self.weight_scale


# Exemple d'utilisation QAT
class QATModel(nn.Module):
    """Modèle simple avec QAT"""

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = QuantizedLinear(input_dim, hidden_dim, n_bits=8)
        self.fc2 = QuantizedLinear(hidden_dim, output_dim, n_bits=8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Training loop avec QAT
def train_qat_model(model, train_loader, epochs=10, lr=1e-3):
    """Entraîner modèle avec Quantization-Aware Training"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} terminée. Loss moyenne: {avg_loss:.4f}")

    # Après training, convertir en vraie quantization
    print("\nConversion en INT8...")
    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                weight_int8, scale = module.convert_to_quantized()
                print(f"{name}: converti en INT8, scale={scale:.6f}")

    return model


# Test
if __name__ == "__main__":
    # Créer modèle QAT
    model = QATModel(input_dim=784, hidden_dim=256, output_dim=10)

    # Données dummy pour démonstration
    train_data = torch.randn(1000, 784)
    train_labels = torch.randint(0, 10, (1000,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training
    model_trained = train_qat_model(model, train_loader, epochs=3, lr=1e-3)

    print("\nQAT training terminé!")
```

**Avantages QAT** :
- Meilleure précision que PTQ (modèle s'adapte)
- Possible de quantizer en INT4 avec perte minimale

**Inconvénients QAT** :
- Nécessite ré-entraînement (coûteux)
- Besoin de dataset complet d'entraînement

---

## 4. Post-Training Quantization (PTQ) Détaillée

### 4.1 PyTorch Quantization API

PyTorch offre plusieurs backends de quantization :

| Backend    | Hardware   | Support                    | Performance |
|------------|------------|----------------------------|-------------|
| `fbgemm`   | x86 CPU    | INT8 (AVX2/AVX512)        | Excellent   |
| `qnnpack`  | ARM CPU    | INT8 (mobile)             | Excellent   |
| `onednn`   | x86 CPU/GPU| INT8, BF16                | Bon         |

### 4.2 Implémentation PTQ Complète

```python
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub

class ModelToQuantize(nn.Module):
    """Modèle exemple pour quantization"""

    def __init__(self):
        super().__init__()
        # Quantization stubs (marqueurs début/fin quantization)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # Quantization stub (input)
        x = self.quant(x)

        # Convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Classifier
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # Dequantization stub (output)
        x = self.dequant(x)

        return x


def apply_ptq_static(model, calibration_loader):
    """
    Appliquer Static Post-Training Quantization

    Args:
        model: Modèle FP32 à quantizer
        calibration_loader: DataLoader pour calibration

    Returns:
        model_int8: Modèle quantizé en INT8
    """

    # 1. Fuse modules (Conv+BN+ReLU → ConvBnReLU pour meilleure quantization)
    model.eval()
    model_fused = quant.fuse_modules(model, [
        ['conv1', 'bn1', 'relu1'],
        ['conv2', 'bn2', 'relu2']
    ])

    # 2. Spécifier qconfig (quantization configuration)
    # 'fbgemm' pour x86 CPU, 'qnnpack' pour ARM
    model_fused.qconfig = quant.get_default_qconfig('fbgemm')

    # 3. Prepare model pour calibration (insère observers)
    model_prepared = quant.prepare(model_fused, inplace=False)

    # 4. Calibration pass (collecte statistiques activations)
    print("Calibration en cours...")
    model_prepared.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            model_prepared(data)
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(calibration_loader)}")

    # 5. Convertir en INT8
    model_int8 = quant.convert(model_prepared, inplace=False)

    print("Quantization terminée!")
    return model_int8


def apply_ptq_dynamic(model):
    """
    Appliquer Dynamic Post-Training Quantization
    (Poids quantizés, activations quantizées dynamiquement)

    Args:
        model: Modèle FP32

    Returns:
        model_int8: Modèle quantizé
    """

    model.eval()

    # Quantize poids en INT8, activations restent FP32 mais quantizées à la volée
    model_int8 = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # Layers à quantizer
        dtype=torch.qint8
    )

    return model_int8


def benchmark_quantization(model_fp32, model_int8, test_loader):
    """
    Benchmarker modèle FP32 vs INT8

    Args:
        model_fp32: Modèle en FP32
        model_int8: Modèle quantizé
        test_loader: DataLoader test

    Returns:
        results: Dictionnaire des résultats
    """
    import time

    device = 'cpu'  # Quantization INT8 pour CPU
    model_fp32 = model_fp32.to(device)
    model_int8 = model_int8.to(device)

    model_fp32.eval()
    model_int8.eval()

    # Benchmark FP32
    print("Benchmark FP32...")
    start = time.time()
    correct_fp32 = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model_fp32(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_fp32 += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    time_fp32 = time.time() - start
    acc_fp32 = 100. * correct_fp32 / total

    # Benchmark INT8
    print("Benchmark INT8...")
    start = time.time()
    correct_int8 = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model_int8(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_int8 += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    time_int8 = time.time() - start
    acc_int8 = 100. * correct_int8 / total

    # Taille mémoire
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.pth")
        size = os.path.getsize("temp.pth") / 1e6  # MB
        os.remove("temp.pth")
        return size

    size_fp32 = get_model_size(model_fp32)
    size_int8 = get_model_size(model_int8)

    # Résultats
    results = {
        'fp32': {
            'accuracy': acc_fp32,
            'time': time_fp32,
            'size_mb': size_fp32
        },
        'int8': {
            'accuracy': acc_int8,
            'time': time_int8,
            'size_mb': size_int8
        },
        'speedup': time_fp32 / time_int8,
        'compression': size_fp32 / size_int8,
        'accuracy_loss': acc_fp32 - acc_int8
    }

    print("\n" + "=" * 60)
    print("RÉSULTATS BENCHMARK")
    print("=" * 60)
    print(f"\nFP32:")
    print(f"  Accuracy: {acc_fp32:.2f}%")
    print(f"  Time: {time_fp32:.2f}s")
    print(f"  Size: {size_fp32:.2f} MB")

    print(f"\nINT8:")
    print(f"  Accuracy: {acc_int8:.2f}%")
    print(f"  Time: {time_int8:.2f}s")
    print(f"  Size: {size_int8:.2f} MB")

    print(f"\nGains:")
    print(f"  Speedup: {results['speedup']:.2f}×")
    print(f"  Compression: {results['compression']:.2f}×")
    print(f"  Accuracy loss: {results['accuracy_loss']:.2f}%")

    return results


# Test complet
if __name__ == "__main__":
    import os

    # Créer modèle FP32
    model_fp32 = ModelToQuantize()

    # Données dummy pour calibration et test
    calibration_data = torch.randn(100, 3, 32, 32)
    calibration_labels = torch.randint(0, 10, (100,))
    calibration_dataset = torch.utils.data.TensorDataset(calibration_data, calibration_labels)
    calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=10)

    test_data = torch.randn(200, 3, 32, 32)
    test_labels = torch.randint(0, 10, (200,))
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20)

    # Appliquer PTQ Static
    model_int8_static = apply_ptq_static(model_fp32, calibration_loader)

    # Benchmark
    results = benchmark_quantization(model_fp32, model_int8_static, test_loader)
```

**Output attendu** :
```
Calibration en cours...
  Batch 0/10
  Batch 10/10
Quantization terminée!
Benchmark FP32...
Benchmark INT8...

============================================================
RÉSULTATS BENCHMARK
============================================================

FP32:
  Accuracy: 12.00%
  Time: 0.45s
  Size: 0.52 MB

INT8:
  Accuracy: 11.50%
  Time: 0.18s
  Size: 0.14 MB

Gains:
  Speedup: 2.50×
  Compression: 3.71×
  Accuracy loss: 0.50%
```

**Conclusion PTQ** : Gain 2-3× performance et 4× compression avec perte accuracy <1%.

---

## 5. Quantization-Aware Training (QAT) Détaillée

### 5.1 QAT avec PyTorch

```python
import torch
import torch.nn as nn
import torch.quantization as quant

class QATModel(nn.Module):
    """Modèle pour Quantization-Aware Training"""

    def __init__(self, num_classes=10):
        super().__init__()
        # Quantization stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # Network
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv+BN+ReLU pour quantization"""
        quant.fuse_modules(self, [
            ['conv1', 'bn1', 'relu1'],
            ['conv2', 'bn2', 'relu2']
        ], inplace=True)


def train_qat(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """
    Entraîner modèle avec Quantization-Aware Training

    Args:
        model: Modèle FP32
        train_loader: DataLoader entraînement
        val_loader: DataLoader validation
        epochs: Nombre d'époques
        lr: Learning rate

    Returns:
        model_qat: Modèle quantizé
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Fuse modules
    model.fuse_model()

    # 2. Spécifier qconfig pour QAT
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')

    # 3. Prepare pour QAT (insère fake quantization)
    model_prepared = quant.prepare_qat(model, inplace=False)
    model_prepared = model_prepared.to(device)

    # 4. Training avec fake quantization
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Démarrage QAT training...")
    for epoch in range(epochs):
        # Training
        model_prepared.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model_prepared(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        train_acc = 100. * correct / total

        # Validation
        model_prepared.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model_prepared(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        val_acc = 100. * correct / total

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")

        # Disable observer après quelques époques (freeze scales)
        if epoch >= epochs // 2:
            model_prepared.apply(quant.disable_observer)

        # Freeze batch norm après la majorité du training
        if epoch >= epochs * 3 // 4:
            model_prepared.apply(nn.intrinsic.qat.freeze_bn_stats)

    # 5. Convertir en INT8 réel
    model_prepared.eval()
    model_prepared = model_prepared.to('cpu')  # Quantized models sur CPU
    model_qat = quant.convert(model_prepared, inplace=False)

    print("\nQAT training terminé. Modèle converti en INT8.")

    return model_qat


# Exemple complet
if __name__ == "__main__":
    # Créer modèle
    model = QATModel(num_classes=10)

    # Données dummy
    train_data = torch.randn(500, 3, 32, 32)
    train_labels = torch.randint(0, 10, (500,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_data = torch.randn(100, 3, 32, 32)
    val_labels = torch.randint(0, 10, (100,))
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # QAT Training
    model_qat = train_qat(model, train_loader, val_loader, epochs=5, lr=1e-3)

    print("\nQAT terminé!")
```

**Output attendu** :
```
Démarrage QAT training...
Epoch 1/5: Train Loss=2.3145, Acc=11.20% | Val Loss=2.2987, Acc=12.00%
Epoch 2/5: Train Loss=2.2876, Acc=13.40% | Val Loss=2.2654, Acc=15.00%
Epoch 3/5: Train Loss=2.2543, Acc=16.20% | Val Loss=2.2321, Acc=17.00%
Epoch 4/5: Train Loss=2.2287, Acc=18.60% | Val Loss=2.2154, Acc=19.00%
Epoch 5/5: Train Loss=2.2098, Acc=20.40% | Val Loss=2.1987, Acc=21.00%

QAT training terminé. Modèle converti en INT8.
```

---

## 6. GPTQ - GPU Post-Training Quantization

### 6.1 Introduction à GPTQ

**GPTQ** (Frantar et al., 2022) est une méthode avancée de quantization pour LLMs qui utilise une approximation de second ordre pour minimiser l'erreur de quantization.

**Problème** : Quantization naïve (round to nearest) ne minimise pas l'erreur de reconstruction pour les poids avec corrélations.

**Solution GPTQ** : Utilise une approximation de la **matrice Hessienne inverse** pour optimiser la quantization layer-par-layer.

### 6.2 Formulation Mathématique

Pour chaque layer, on cherche à minimiser :

$$
\arg\min_{\hat{W}} \|WX - \hat{W}X\|_2^2
$$

Où :
- $W$ : poids originaux (FP16)
- $\hat{W}$ : poids quantizés (INT4/INT3)
- $X$ : activations calibration

**GPTQ utilise l'approximation OBQ (Optimal Brain Quantization)** :

$$
\delta F_q = \frac{(w_q - w)^2}{[H^{-1}]_{qq}}
$$

Où $H$ est la Hessienne de la loss par rapport aux poids.

**Algorithme GPTQ simplifié** :

```
Pour chaque layer:
    1. Calculer Hessienne inverse H^-1 (approximation)
    2. Pour chaque colonne j des poids:
        a. Quantize w_j
        b. Calculer erreur e_j = w_j - w_j_quant
        c. Compenser erreur sur poids restants:
           W[:, j+1:] -= e_j * (H^-1[j, j+1:] / H^-1[j,j])
```

Cette compensation réduit drastiquement l'erreur de reconstruction.

### 6.3 Implémentation GPTQ avec AutoGPTQ

**Installation** :
```bash
pip install auto-gptq transformers accelerate
```

**Code complet** :

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def quantize_model_gptq(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    dataset: str = "c4",
    num_samples: int = 128
):
    """
    Quantize un LLM avec GPTQ

    Args:
        model_name: Nom du modèle HuggingFace (e.g., "meta-llama/Llama-2-7b-hf")
        output_dir: Répertoire de sauvegarde
        bits: Nombre de bits (2, 3, 4, 8)
        group_size: Group size pour quantization (128 recommandé)
        dataset: Dataset calibration ("c4", "wikitext2", "ptb")
        num_samples: Nombre d'exemples calibration

    Returns:
        model_gptq: Modèle quantizé
    """

    print(f"Chargement du modèle: {model_name}")

    # Configuration quantization
    quantize_config = BaseQuantizeConfig(
        bits=bits,                    # Bits de quantization
        group_size=group_size,        # Group size (128 ou -1 pour per-channel)
        desc_act=False,               # Order of activations (False plus rapide)
        damp_percent=0.01,           # Damping pour stabilité Hessienne
        sym=True,                     # Symmetric quantization
        true_sequential=True          # Quantize séquentiellement (plus précis)
    )

    # Charger modèle
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        device_map="auto"             # Automatic device placement
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Préparer dataset de calibration
    print(f"Préparation dataset calibration: {dataset}")
    from datasets import load_dataset

    if dataset == "c4":
        data = load_dataset("allenai/c4", "en", split="train", streaming=True)
        data = data.shuffle(seed=42).take(num_samples)
    elif dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        raise ValueError(f"Dataset {dataset} non supporté")

    # Tokenize calibration data
    calibration_data = []
    for example in data:
        text = example["text"] if "text" in example else example["content"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        )
        calibration_data.append(tokens)

        if len(calibration_data) >= num_samples:
            break

    print(f"Dataset calibration: {len(calibration_data)} exemples")

    # Quantization avec GPTQ
    print(f"Quantization GPTQ en {bits} bits...")
    model.quantize(
        calibration_data,
        use_triton=False,  # Triton kernel (plus rapide sur GPU Ampere+)
        batch_size=1
    )

    # Sauvegarder modèle quantizé
    print(f"Sauvegarde dans {output_dir}")
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)

    print("GPTQ quantization terminée!")
    return model


def load_and_test_gptq(model_dir: str):
    """
    Charger et tester un modèle GPTQ quantizé

    Args:
        model_dir: Répertoire du modèle quantizé
    """

    print(f"Chargement modèle GPTQ: {model_dir}")

    # Charger modèle quantizé
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        device_map="auto",
        use_triton=False,
        use_safetensors=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Test génération
    prompt = "La quantization des modèles de langage permet de"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}")
    print("Génération...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nRésultat:\n{generated_text}")

    # Benchmark memory
    print(f"\nMémoire GPU utilisée: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model


# Exemple complet
if __name__ == "__main__":
    # Quantize Llama 2 7B en 4-bit avec GPTQ
    model_name = "meta-llama/Llama-2-7b-hf"  # Ou autre modèle
    output_dir = "./llama-2-7b-gptq-4bit"

    # Quantization (nécessite 1× GPU avec ~16GB VRAM)
    model_gptq = quantize_model_gptq(
        model_name=model_name,
        output_dir=output_dir,
        bits=4,
        group_size=128,
        dataset="c4",
        num_samples=128
    )

    # Test du modèle quantizé
    model = load_and_test_gptq(output_dir)

    print("\nGPTQ quantization et test terminés!")
```

**Output attendu** :
```
Chargement du modèle: meta-llama/Llama-2-7b-hf
Préparation dataset calibration: c4
Dataset calibration: 128 exemples
Quantization GPTQ en 4 bits...
Layer 0/32...
Layer 10/32...
Layer 20/32...
Layer 32/32...
Sauvegarde dans ./llama-2-7b-gptq-4bit
GPTQ quantization terminée!

Chargement modèle GPTQ: ./llama-2-7b-gptq-4bit
Prompt: La quantization des modèles de langage permet de
Génération...

Résultat:
La quantization des modèles de langage permet de réduire significativement
l'empreinte mémoire et d'accélérer l'inférence, tout en préservant la qualité
des générations. Cette technique transforme les poids en précision réduite...

Mémoire GPU utilisée: 3.82 GB
```

**Gains GPTQ** :
- **Llama 2 7B** : 14GB (FP16) → 3.5GB (INT4) = **4× compression**
- **Perte accuracy** : <2% sur benchmarks
- **Speedup** : 2-3× sur génération

### 6.4 Comparaison Group Size

```python
def compare_gptq_group_sizes(model_name: str):
    """Comparer différents group sizes GPTQ"""

    group_sizes = [128, 64, 32, -1]  # -1 = per-channel
    results = {}

    for gs in group_sizes:
        print(f"\n{'='*60}")
        print(f"Testing group_size={gs}")
        print('='*60)

        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=gs,
            desc_act=False,
            sym=True
        )

        # Quantization (simplifié pour benchmark)
        model = AutoGPTQForCausalLM.from_pretrained(
            model_name,
            quantize_config=quantize_config
        )

        # Calculer taille mémoire
        param_count = sum(p.numel() for p in model.parameters())
        # INT4 = 0.5 bytes/param + overhead scales
        if gs == -1:
            # Per-channel: 1 scale par output channel
            num_scales = param_count // 1000  # Approximation
        else:
            # Per-group: 1 scale par groupe
            num_scales = param_count // gs

        total_bytes = (param_count * 0.5) + (num_scales * 2)  # FP16 scales
        total_mb = total_bytes / 1e6

        results[gs] = {
            'size_mb': total_mb,
            'num_scales': num_scales
        }

        print(f"Taille modèle: {total_mb:.2f} MB")
        print(f"Nombre de scales: {num_scales}")

    # Comparaison
    print("\n" + "="*60)
    print("COMPARAISON GROUP SIZES")
    print("="*60)
    print(f"{'Group Size':<15} {'Taille (MB)':<15} {'# Scales':<15} {'Qualité'}")
    print("-"*60)

    quality_map = {128: "Bonne", 64: "Excellente", 32: "Meilleure", -1: "Optimale"}
    for gs in group_sizes:
        print(f"{str(gs):<15} {results[gs]['size_mb']:<15.2f} "
              f"{results[gs]['num_scales']:<15} {quality_map[gs]}")

# Exécution
# compare_gptq_group_sizes("meta-llama/Llama-2-7b-hf")
```

**Trade-off Group Size** :
- **group_size=128** : Bon compromis (recommandé)
- **group_size=64** : Meilleure précision, +overhead
- **group_size=-1** : Per-channel (optimal mais plus lent)

---

## 7. AWQ - Activation-aware Weight Quantization

### 7.1 Introduction à AWQ

**AWQ** (Lin et al., 2023) est une méthode récente qui observe que **tous les poids ne sont pas égaux** : certains canaux (salient channels) ont un impact disproportionné sur la performance.

**Observation clé** : 1% des canaux peuvent contenir 99% de l'information critique.

**Stratégie AWQ** :
1. Identifier canaux saillants (via magnitude activations)
2. Protéger ces canaux avec scaling avant quantization
3. Quantizer tous les poids en INT4 (pas de mixed-precision)

### 7.2 Formulation Mathématique

AWQ optimise la quantization avec scaling per-channel :

$$
\mathbf{Q}(W \cdot s) \cdot (X / s)
$$

Où :
- $s$ : scaling factors (learnable)
- Appliqué sur **poids** : $W \cdot s$
- Inverse sur **activations** : $X / s$

Les scaling factors $s$ sont optimisés pour minimiser :

$$
\mathcal{L}(\mathbf{s}) = \|W \cdot s \cdot X / s - \mathbf{Q}(W \cdot s) \cdot X / s\|
$$

**Recherche grid** pour trouver scales optimaux (rapide, ~min).

### 7.3 Implémentation AWQ

**Installation** :
```bash
pip install autoawq accelerate
```

**Code complet** :

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_model_awq(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True
):
    """
    Quantize un LLM avec AWQ

    Args:
        model_name: Nom du modèle HuggingFace
        output_dir: Répertoire de sauvegarde
        bits: Nombre de bits (4 recommandé)
        group_size: Group size (128 optimal)
        zero_point: Utiliser zero-point (True recommandé)

    Returns:
        model: Modèle quantizé
    """

    print(f"Chargement modèle: {model_name}")

    # Charger modèle
    model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configuration quantization
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM"  # Kernel (GEMM ou GEMV)
    }

    # Dataset calibration (AWQ nécessite moins d'exemples que GPTQ)
    print("Préparation calibration data...")
    from datasets import load_dataset

    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    data = data.shuffle(seed=42).take(128)

    calibration_data = []
    for example in data:
        text = example["text"]
        calibration_data.append(text)

    # Quantization AWQ
    print(f"Quantization AWQ {bits}-bit...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_data
    )

    # Sauvegarder
    print(f"Sauvegarde dans {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("AWQ quantization terminée!")
    return model


def load_and_benchmark_awq(model_dir: str):
    """Charger et benchmarker modèle AWQ"""

    print(f"Chargement modèle AWQ: {model_dir}")

    # Charger modèle quantizé
    model = AutoAWQForCausalLM.from_quantized(
        model_dir,
        fuse_layers=True,  # Fuse layers pour meilleure performance
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Benchmark latence
    import time

    prompt = "Explain quantization in one sentence:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(3):
        _ = model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    num_runs = 20
    latencies = []

    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        latencies.append(time.time() - start)

    avg_latency = sum(latencies) / len(latencies)
    throughput = 50 / avg_latency  # tokens/sec

    print(f"\n{'='*60}")
    print("BENCHMARK AWQ")
    print('='*60)
    print(f"Latence moyenne: {avg_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"Mémoire GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Test génération
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGénération:\n{generated}")

    return {
        'latency_ms': avg_latency * 1000,
        'throughput': throughput,
        'memory_gb': torch.cuda.memory_allocated() / 1e9
    }


# Exemple complet
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    output_dir = "./llama-2-7b-awq-4bit"

    # Quantization
    model = quantize_model_awq(
        model_name=model_name,
        output_dir=output_dir,
        bits=4,
        group_size=128
    )

    # Benchmark
    results = load_and_benchmark_awq(output_dir)

    print("\nAWQ quantization et benchmark terminés!")
```

**Output attendu** :
```
Chargement modèle: meta-llama/Llama-2-7b-hf
Préparation calibration data...
Quantization AWQ 4-bit...
Searching best scales... (Layer 0/32)
Searching best scales... (Layer 32/32)
Sauvegarde dans ./llama-2-7b-awq-4bit
AWQ quantization terminée!

Chargement modèle AWQ: ./llama-2-7b-awq-4bit

============================================================
BENCHMARK AWQ
============================================================
Latence moyenne: 245.32 ms
Throughput: 203.85 tokens/sec
Mémoire GPU: 3.65 GB

Génération:
Explain quantization in one sentence: Quantization reduces the precision
of model weights and activations to lower bit representations, enabling
smaller models and faster inference while maintaining accuracy.

AWQ quantization et benchmark terminés!
```

### 7.4 AWQ vs GPTQ Comparaison

```python
def compare_awq_vs_gptq():
    """Tableau comparatif AWQ vs GPTQ"""

    comparison = {
        'Méthode': ['AWQ', 'GPTQ'],
        'Approche': [
            'Activation-aware scaling',
            'Hessienne inverse (OBQ)'
        ],
        'Calibration samples': ['128', '128-512'],
        'Temps quantization': ['~5-10 min', '~15-30 min'],
        'Qualité (perplexity)': ['Excellente', 'Très bonne'],
        'Vitesse inference': ['Très rapide', 'Rapide'],
        'Support INT3': ['Non', 'Oui'],
        'Support INT4': ['Oui', 'Oui'],
        'Memory overhead': ['Minimal', 'Faible']
    }

    import pandas as pd
    df = pd.DataFrame(comparison)

    print("\n" + "="*80)
    print("COMPARAISON AWQ vs GPTQ")
    print("="*80)
    print(df.to_string(index=False))
    print("\n")

    # Recommendations
    print("RECOMMANDATIONS:")
    print("  • AWQ: Meilleur pour inference rapide (production)")
    print("  • GPTQ: Meilleur pour compression maximale (INT3/INT2)")
    print("  • Les deux: Excellente qualité en INT4")

# Exécution
compare_awq_vs_gptq()
```

**Output** :
```
================================================================================
COMPARAISON AWQ vs GPTQ
================================================================================
 Méthode                      Approche  Calibration samples Temps quantization         Qualité (perplexity) Vitesse inference Support INT3 Support INT4 Memory overhead
     AWQ  Activation-aware scaling                  128        ~5-10 min                   Excellente       Très rapide          Non          Oui         Minimal
    GPTQ  Hessienne inverse (OBQ)               128-512       ~15-30 min                   Très bonne            Rapide          Oui          Oui          Faible


RECOMMANDATIONS:
  • AWQ: Meilleur pour inference rapide (production)
  • GPTQ: Meilleur pour compression maximale (INT3/INT2)
  • Les deux: Excellente qualité en INT4
```

---

## 8. GGUF et llama.cpp

### 8.1 Introduction à GGUF

**GGUF** (GPT-Generated Unified Format) est le format de quantization utilisé par **llama.cpp**, permettant d'exécuter des LLMs sur **CPU** avec performance remarquable.

**Avantages GGUF/llama.cpp** :
- **Inference CPU pure** (pas de GPU nécessaire)
- **Support K-quantization** (mélanges INT4/INT5/INT6 stratégiques)
- **Memory mapping** efficace
- **Multi-plateforme** (Windows, macOS, Linux, mobile)

**Use case** : Déploiement sur hardware sans GPU (edge devices, laptops).

### 8.2 Types de Quantization GGUF

GGUF offre de nombreux formats de quantization :

| Format   | Bits  | Description                           | Qualité | Taille (7B) |
|----------|-------|---------------------------------------|---------|-------------|
| F16      | 16    | Full half-precision                   | 100%    | 14 GB       |
| Q8_0     | 8     | INT8, fast                            | 99%     | 7 GB        |
| Q6_K     | 6.5   | Mixed 6-bit (quality)                 | 98%     | 5.5 GB      |
| Q5_K_M   | 5.5   | Mixed 5-bit (medium)                  | 96%     | 4.8 GB      |
| Q4_K_M   | 4.5   | Mixed 4-bit (medium, recommandé)      | 93%     | 4.1 GB      |
| Q4_0     | 4     | INT4, smallest                        | 90%     | 3.5 GB      |
| Q3_K_M   | 3.5   | Mixed 3-bit (extreme compression)     | 85%     | 3.0 GB      |
| Q2_K     | 2.5   | Mixed 2-bit (experimental)            | 75%     | 2.5 GB      |

**K-quantization** : Utilise différents bits par layer (attention Q6, FFN Q4, etc.) pour optimiser qualité/taille.

### 8.3 Conversion et Utilisation llama.cpp

**Installation llama.cpp** :
```bash
# Cloner repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Compiler (avec support CUDA optionnel)
make  # CPU only
# OU
make LLAMA_CUBLAS=1  # Avec CUDA

# Compiler avec support Metal (macOS)
make LLAMA_METAL=1
```

**Conversion modèle HuggingFace → GGUF** :

```bash
# Installation dépendances Python
pip install -r requirements.txt

# Convertir modèle (ex: Llama 2 7B)
python convert.py /path/to/llama-2-7b-hf \
    --outtype f16 \
    --outfile llama-2-7b-f16.gguf

# Quantizer en différents formats
./quantize llama-2-7b-f16.gguf llama-2-7b-Q4_K_M.gguf Q4_K_M
./quantize llama-2-7b-f16.gguf llama-2-7b-Q5_K_M.gguf Q5_K_M
./quantize llama-2-7b-f16.gguf llama-2-7b-Q8_0.gguf Q8_0
```

**Inference avec llama.cpp** :

```bash
# Charger modèle et générer
./main -m llama-2-7b-Q4_K_M.gguf \
    -p "Explain quantization in simple terms:" \
    -n 256 \
    --temp 0.7 \
    --top-p 0.9 \
    --ctx-size 2048 \
    --threads 8

# Serveur API (compatible OpenAI)
./server -m llama-2-7b-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 32  # Offload layers vers GPU si disponible
```

### 8.4 Utilisation Python avec llama-cpp-python

**Installation** :
```bash
pip install llama-cpp-python

# Avec support CUDA (optionnel)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

**Code Python** :

```python
from llama_cpp import Llama

def load_and_run_gguf(model_path: str):
    """
    Charger et exécuter un modèle GGUF avec llama-cpp-python

    Args:
        model_path: Chemin vers fichier .gguf
    """

    print(f"Chargement modèle GGUF: {model_path}")

    # Charger modèle
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,           # Context window
        n_threads=8,          # CPU threads
        n_gpu_layers=0,       # Layers sur GPU (0 = CPU only)
        verbose=False
    )

    # Test génération
    prompt = "Explain quantization in one paragraph:"

    print(f"\nPrompt: {prompt}")
    print("Génération...\n")

    # Génération avec streaming
    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        echo=False,           # Ne pas répéter le prompt
        stream=True           # Streaming (affichage token-par-token)
    )

    # Afficher tokens au fur et à mesure
    full_response = ""
    for chunk in output:
        token = chunk['choices'][0]['text']
        full_response += token
        print(token, end='', flush=True)

    print("\n")

    # Benchmark
    import time

    print("\nBenchmark...")
    prompts = [
        "What is AI?",
        "Explain machine learning:",
        "Describe neural networks:"
    ]

    total_tokens = 0
    start = time.time()

    for p in prompts:
        output = llm(p, max_tokens=50, stream=False)
        total_tokens += len(llm.tokenize(output['choices'][0]['text'].encode()))

    elapsed = time.time() - start
    throughput = total_tokens / elapsed

    print(f"Tokens générés: {total_tokens}")
    print(f"Temps total: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/sec")

    return llm


# Exemple
if __name__ == "__main__":
    model_path = "./llama-2-7b-Q4_K_M.gguf"
    llm = load_and_run_gguf(model_path)

    print("\nGGUF inference terminée!")
```

**Output attendu** :
```
Chargement modèle GGUF: ./llama-2-7b-Q4_K_M.gguf

Prompt: Explain quantization in one paragraph:
Génération...

Quantization is a technique used to reduce the precision of numerical
representations in machine learning models, particularly large language
models. By converting high-precision floating-point weights (like FP32 or
FP16) to lower-precision integers (like INT8 or INT4), quantization
significantly reduces model size and memory requirements. This compression
enables deployment on resource-constrained devices while maintaining
acceptable performance, though with a small trade-off in accuracy.

Benchmark...
Tokens générés: 150
Temps total: 5.42s
Throughput: 27.68 tokens/sec

GGUF inference terminée!
```

**Performance llama.cpp** :
- **CPU (Apple M1)** : 20-30 tokens/sec (7B Q4)
- **CPU (AMD Ryzen 9)** : 15-25 tokens/sec (7B Q4)
- **GPU offload** : 60-100 tokens/sec (avec --n-gpu-layers)

### 8.5 Comparaison Quantization Types

```python
import os
import subprocess

def benchmark_gguf_quantizations(base_model: str):
    """
    Benchmarker différents types de quantization GGUF

    Args:
        base_model: Modèle HuggingFace de base
    """

    quant_types = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M"]

    results = []

    # Convertir base model en F16
    print("Conversion en F16...")
    subprocess.run([
        "python", "llama.cpp/convert.py", base_model,
        "--outtype", "f16",
        "--outfile", "model-f16.gguf"
    ])

    # Quantize vers chaque type
    for qtype in quant_types:
        print(f"\n{'='*60}")
        print(f"Quantization: {qtype}")
        print('='*60)

        output_file = f"model-{qtype}.gguf"

        # Quantize
        subprocess.run([
            "./llama.cpp/quantize",
            "model-f16.gguf",
            output_file,
            qtype
        ])

        # Mesurer taille
        size_mb = os.path.getsize(output_file) / 1e6

        # Benchmark perplexity (optionnel, sur dataset test)
        # perplexity = run_perplexity_test(output_file)

        results.append({
            'type': qtype,
            'size_mb': size_mb,
            # 'perplexity': perplexity
        })

        print(f"Taille: {size_mb:.2f} MB")

    # Afficher résultats
    print("\n" + "="*60)
    print("RÉSUMÉ QUANTIZATIONS GGUF")
    print("="*60)
    print(f"{'Type':<12} {'Taille (MB)':<15} {'Compression':<15}")
    print("-"*60)

    base_size = results[0]['size_mb']  # Q8_0 comme référence
    for r in results:
        compression = base_size / r['size_mb']
        print(f"{r['type']:<12} {r['size_mb']:<15.2f} {compression:<15.2f}×")

    print("\nRecommandation: Q4_K_M pour meilleur compromis qualité/taille")

# Exécution
# benchmark_gguf_quantizations("meta-llama/Llama-2-7b-hf")
```

---

## 9. BitsAndBytes Integration

### 9.1 Introduction à BitsAndBytes

**BitsAndBytes** est une librairie optimisée pour quantization INT8/INT4 avec support **CUDA kernels** et intégration transparente avec HuggingFace Transformers.

**Features clés** :
- **LLM.int8()** : Quantization INT8 avec mixed-precision pour outliers
- **NF4 quantization** : Normalfloat 4-bit (utilisé par QLoRA)
- **Double quantization** : Quantize les scales eux-mêmes
- **Paged optimizers** : Gestion mémoire optimisée

### 9.2 LLM.int8() - Quantization INT8

**Problème des outliers** : Certaines features (0.1% des valeurs) ont magnitude 100× supérieure, causant dégradation si quantizées naïvement.

**Solution LLM.int8()** (Dettmers et al., 2022) :
- Détecter outliers (|x| > threshold)
- Traiter outliers en FP16 (mixed-precision)
- Quantizer reste en INT8

**Implémentation** :

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_int8(model_name: str):
    """
    Charger modèle en INT8 avec BitsAndBytes (LLM.int8())

    Args:
        model_name: Nom du modèle HuggingFace

    Returns:
        model: Modèle quantizé INT8
    """

    print(f"Chargement modèle INT8: {model_name}")

    # Configuration INT8
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,       # Threshold pour outliers
        llm_int8_has_fp16_weight=False  # Poids stockés en INT8
    )

    # Charger modèle
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",            # Automatic device placement
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Modèle chargé. Mémoire GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer


def test_int8_generation(model, tokenizer):
    """Tester génération avec modèle INT8"""

    prompt = "The key advantages of 8-bit quantization are:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}")
    print("Génération...\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)


# Exemple
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"

    # Charger en INT8
    model_int8, tokenizer = load_model_int8(model_name)

    # Test génération
    test_int8_generation(model_int8, tokenizer)

    # Comparaison mémoire avec FP16
    print("\n" + "="*60)
    print("COMPARAISON MÉMOIRE")
    print("="*60)
    print(f"FP16 (baseline): ~14 GB")
    print(f"INT8 (actual): {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Compression: {14 / (torch.cuda.memory_allocated() / 1e9):.2f}×")
```

**Output attendu** :
```
Chargement modèle INT8: meta-llama/Llama-2-7b-hf
Modèle chargé. Mémoire GPU: 7.12 GB

Prompt: The key advantages of 8-bit quantization are:
Génération...

The key advantages of 8-bit quantization are: reduced memory footprint
(~2× smaller), faster inference on integer operations, lower power
consumption, and easier deployment on resource-constrained devices. The
LLM.int8() technique maintains accuracy by handling outlier features
separately in FP16 precision.

============================================================
COMPARAISON MÉMOIRE
============================================================
FP16 (baseline): ~14 GB
INT8 (actual): 7.12 GB
Compression: 1.97×
```

### 9.3 NF4 Quantization (QLoRA)

NF4 (NormalFloat 4-bit) est optimal pour poids qui suivent une distribution normale.

**Voir Chapitre 13 : LoRA & QLoRA** pour implémentation complète NF4 + double quantization.

**Rappel code NF4** :

```python
# Configuration NF4 (4-bit)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat 4-bit
    bnb_4bit_use_double_quant=True,   # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Charger modèle
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Résultat: Llama 2 7B en ~4GB au lieu de 14GB!
```

---

## 10. Benchmarks et Comparaisons

### 10.1 Benchmark Complet: Toutes les Méthodes

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd

def comprehensive_quantization_benchmark(model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Benchmark complet de toutes les méthodes de quantization

    Compare: FP16, INT8, NF4, GPTQ-4bit, AWQ-4bit, GGUF-Q4
    """

    results = []
    test_prompt = "Explain artificial intelligence:"

    # 1. FP16 Baseline
    print("\n" + "="*70)
    print("1. FP16 BASELINE")
    print("="*70)

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    memory_fp16 = torch.cuda.memory_allocated() / 1e9
    latency_fp16 = benchmark_latency(model_fp16, tokenizer, test_prompt)

    results.append({
        'Method': 'FP16',
        'Memory (GB)': memory_fp16,
        'Latency (ms)': latency_fp16,
        'Throughput (tok/s)': 50 / (latency_fp16 / 1000),
        'Compression': 1.0
    })

    print(f"Memory: {memory_fp16:.2f} GB")
    print(f"Latency: {latency_fp16:.2f} ms")

    del model_fp16
    torch.cuda.empty_cache()

    # 2. INT8 (BitsAndBytes)
    print("\n" + "="*70)
    print("2. INT8 (BitsAndBytes LLM.int8())")
    print("="*70)

    config_int8 = BitsAndBytesConfig(load_in_8bit=True)
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config_int8,
        device_map="auto"
    )

    memory_int8 = torch.cuda.memory_allocated() / 1e9
    latency_int8 = benchmark_latency(model_int8, tokenizer, test_prompt)

    results.append({
        'Method': 'INT8',
        'Memory (GB)': memory_int8,
        'Latency (ms)': latency_int8,
        'Throughput (tok/s)': 50 / (latency_int8 / 1000),
        'Compression': memory_fp16 / memory_int8
    })

    print(f"Memory: {memory_int8:.2f} GB")
    print(f"Latency: {latency_int8:.2f} ms")

    del model_int8
    torch.cuda.empty_cache()

    # 3. NF4 (QLoRA)
    print("\n" + "="*70)
    print("3. NF4 4-bit (QLoRA)")
    print("="*70)

    config_nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_nf4 = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config_nf4,
        device_map="auto"
    )

    memory_nf4 = torch.cuda.memory_allocated() / 1e9
    latency_nf4 = benchmark_latency(model_nf4, tokenizer, test_prompt)

    results.append({
        'Method': 'NF4',
        'Memory (GB)': memory_nf4,
        'Latency (ms)': latency_nf4,
        'Throughput (tok/s)': 50 / (latency_nf4 / 1000),
        'Compression': memory_fp16 / memory_nf4
    })

    print(f"Memory: {memory_nf4:.2f} GB")
    print(f"Latency: {latency_nf4:.2f} ms")

    del model_nf4
    torch.cuda.empty_cache()

    # 4. GPTQ (si disponible)
    # 5. AWQ (si disponible)
    # 6. GGUF (CPU benchmark séparé)

    # Résultats
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RÉSUMÉ BENCHMARK QUANTIZATION")
    print("="*80)
    print(df.to_string(index=False))
    print("\n")

    # Recommandations
    print("RECOMMANDATIONS:")
    print("  • FP16: Baseline (précision maximale)")
    print("  • INT8: Bon compromis (2× compression, perte minimale)")
    print("  • NF4: Meilleure compression (3-4×, qualité excellente avec QLoRA)")
    print("  • GPTQ/AWQ: Optimal pour production (4× compression, rapide)")
    print("  • GGUF: Déploiement CPU/edge devices")

    return df


def benchmark_latency(model, tokenizer, prompt: str, num_tokens: int = 50, num_runs: int = 10):
    """Mesurer latence moyenne de génération"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(3):
        _ = model.generate(**inputs, max_new_tokens=10)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)  # ms

    return sum(latencies) / len(latencies)


# Exécution
if __name__ == "__main__":
    df_results = comprehensive_quantization_benchmark()

    # Sauvegarder résultats
    df_results.to_csv("quantization_benchmark_results.csv", index=False)
    print("\nRésultats sauvegardés: quantization_benchmark_results.csv")
```

**Output attendu** :
```
==============================================================================
1. FP16 BASELINE
==============================================================================
Memory: 13.48 GB
Latency: 423.56 ms

==============================================================================
2. INT8 (BitsAndBytes LLM.int8())
==============================================================================
Memory: 6.99 GB
Latency: 387.23 ms

==============================================================================
3. NF4 4-bit (QLoRA)
==============================================================================
Memory: 3.82 GB
Latency: 456.78 ms

================================================================================
RÉSUMÉ BENCHMARK QUANTIZATION
================================================================================
 Method  Memory (GB)  Latency (ms)  Throughput (tok/s)  Compression
   FP16        13.48        423.56              118.07         1.00
   INT8         6.99        387.23              129.13         1.93
    NF4         3.82        456.78              109.47         3.53

RECOMMANDATIONS:
  • FP16: Baseline (précision maximale)
  • INT8: Bon compromis (2× compression, perte minimale)
  • NF4: Meilleure compression (3-4×, qualité excellente avec QLoRA)
  • GPTQ/AWQ: Optimal pour production (4× compression, rapide)
  • GGUF: Déploiement CPU/edge devices

Résultats sauvegardés: quantization_benchmark_results.csv
```

### 10.2 Comparaison Qualité (Perplexity)

```python
from datasets import load_dataset
import numpy as np

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", num_samples=100):
    """
    Évaluer perplexity d'un modèle quantizé

    Args:
        model: Modèle à évaluer
        tokenizer: Tokenizer
        dataset_name: Dataset d'évaluation
        num_samples: Nombre d'exemples

    Returns:
        perplexity: Perplexity moyenne
    """

    # Charger dataset
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="test")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for example in dataset:
            text = example["text"]
            if len(text.strip()) < 10:
                continue

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Accumuler
            num_tokens = inputs["input_ids"].numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Calculer perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def compare_quantization_quality(model_name: str):
    """Comparer qualité (perplexity) des différentes quantizations"""

    results = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_configs = {
        'FP16': None,
        'INT8': BitsAndBytesConfig(load_in_8bit=True),
        'NF4': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    }

    for name, config in quantization_configs.items():
        print(f"\nÉvaluation {name}...")

        # Charger modèle
        if config is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config,
                device_map="auto"
            )

        # Évaluer perplexity
        ppl = evaluate_perplexity(model, tokenizer, num_samples=50)

        results.append({
            'Method': name,
            'Perplexity': ppl
        })

        print(f"{name} Perplexity: {ppl:.4f}")

        del model
        torch.cuda.empty_cache()

    # Afficher résultats
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("COMPARAISON QUALITÉ (Perplexity)")
    print("="*50)
    print(df.to_string(index=False))
    print("\nNote: Perplexity plus faible = meilleure qualité")

    return df

# Exécution
# df_quality = compare_quantization_quality("meta-llama/Llama-2-7b-hf")
```

---

## 11. Projet Pratique Complet

### 11.1 Objectif du Projet

Créer un **service d'inference multi-quantization** qui :
1. Charge un modèle avec différentes quantizations
2. Compare performances (latence, mémoire, qualité)
3. Expose une API REST pour génération de texte
4. Permet de sélectionner la quantization selon use case

### 11.2 Architecture du Projet

```
quantization-service/
├── models/
│   ├── loader.py          # Chargement modèles quantizés
│   └── quantizers.py      # Utilitaires quantization
├── api/
│   ├── app.py             # FastAPI application
│   └── schemas.py         # Pydantic models
├── benchmarks/
│   ├── performance.py     # Benchmarks performance
│   └── quality.py         # Évaluation qualité
├── config.py              # Configuration
├── requirements.txt
└── README.md
```

### 11.3 Implémentation Complète

**requirements.txt** :
```
torch>=2.0.0
transformers>=4.35.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
bitsandbytes>=0.41.0
auto-gptq>=0.5.0
autoawq>=0.1.6
accelerate>=0.24.0
datasets>=2.14.0
pandas>=2.0.0
```

**config.py** :
```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    """Configuration du modèle"""
    name: str = "meta-llama/Llama-2-7b-hf"
    cache_dir: str = "./model_cache"
    device_map: str = "auto"

@dataclass
class QuantizationConfig:
    """Configuration de quantization"""
    method: Literal["fp16", "int8", "nf4", "gptq", "awq"]
    bits: int = 4
    group_size: int = 128

@dataclass
class APIConfig:
    """Configuration API"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
```

**models/loader.py** :
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM
from config import ModelConfig, QuantizationConfig

class ModelLoader:
    """Chargeur de modèles avec différentes quantizations"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.models = {}
        self.tokenizer = None

    def load_model(self, quant_config: QuantizationConfig):
        """
        Charger modèle avec quantization spécifiée

        Args:
            quant_config: Configuration de quantization

        Returns:
            model: Modèle chargé
        """

        cache_key = f"{quant_config.method}_{quant_config.bits}bit"

        # Vérifier si déjà chargé
        if cache_key in self.models:
            print(f"Modèle {cache_key} déjà chargé (cache)")
            return self.models[cache_key]

        print(f"Chargement modèle: {quant_config.method} {quant_config.bits}-bit")

        # Charger tokenizer (une fois)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.name,
                cache_dir=self.model_config.cache_dir
            )

        # Charger modèle selon méthode
        if quant_config.method == "fp16":
            model = self._load_fp16()
        elif quant_config.method == "int8":
            model = self._load_int8()
        elif quant_config.method == "nf4":
            model = self._load_nf4()
        elif quant_config.method == "gptq":
            model = self._load_gptq(quant_config.bits, quant_config.group_size)
        elif quant_config.method == "awq":
            model = self._load_awq(quant_config.bits, quant_config.group_size)
        else:
            raise ValueError(f"Méthode {quant_config.method} non supportée")

        # Cache
        self.models[cache_key] = model

        # Log mémoire
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1e9
            print(f"Mémoire GPU utilisée: {memory_gb:.2f} GB")

        return model

    def _load_fp16(self):
        """Charger modèle FP16"""
        return AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            torch_dtype=torch.float16,
            device_map=self.model_config.device_map,
            cache_dir=self.model_config.cache_dir
        )

    def _load_int8(self):
        """Charger modèle INT8 (BitsAndBytes)"""
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            quantization_config=bnb_config,
            device_map=self.model_config.device_map,
            cache_dir=self.model_config.cache_dir
        )

    def _load_nf4(self):
        """Charger modèle NF4 4-bit (QLoRA)"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            quantization_config=bnb_config,
            device_map=self.model_config.device_map,
            cache_dir=self.model_config.cache_dir
        )

    def _load_gptq(self, bits: int, group_size: int):
        """Charger modèle GPTQ quantizé"""
        # Suppose que le modèle GPTQ est pré-quantizé et sauvegardé
        gptq_model_path = f"{self.model_config.cache_dir}/gptq_{bits}bit"

        return AutoGPTQForCausalLM.from_quantized(
            gptq_model_path,
            device_map=self.model_config.device_map,
            use_triton=False
        )

    def _load_awq(self, bits: int, group_size: int):
        """Charger modèle AWQ quantizé"""
        awq_model_path = f"{self.model_config.cache_dir}/awq_{bits}bit"

        return AutoAWQForCausalLM.from_quantized(
            awq_model_path,
            fuse_layers=True,
            device_map=self.model_config.device_map
        )

    def get_tokenizer(self):
        """Retourner tokenizer"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.name,
                cache_dir=self.model_config.cache_dir
            )
        return self.tokenizer

    def clear_cache(self):
        """Vider cache modèles"""
        self.models = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cache modèles vidé")
```

**api/schemas.py** :
```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class GenerationRequest(BaseModel):
    """Requête de génération"""
    prompt: str = Field(..., description="Prompt de génération")
    quantization: Literal["fp16", "int8", "nf4", "gptq", "awq"] = Field(
        default="nf4",
        description="Méthode de quantization"
    )
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    do_sample: bool = Field(default=True)

class GenerationResponse(BaseModel):
    """Réponse de génération"""
    generated_text: str
    quantization_method: str
    num_tokens: int
    latency_ms: float
    memory_gb: Optional[float] = None

class BenchmarkResponse(BaseModel):
    """Réponse benchmark"""
    quantization_method: str
    memory_gb: float
    latency_ms: float
    throughput_tokens_per_sec: float
    compression_ratio: float
```

**api/app.py** :
```python
import torch
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.loader import ModelLoader
from config import ModelConfig, QuantizationConfig, APIConfig
from api.schemas import GenerationRequest, GenerationResponse, BenchmarkResponse

# Initialisation
app = FastAPI(
    title="Quantization Service API",
    description="Service d'inference multi-quantization pour LLMs",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Configuration
model_config = ModelConfig()
api_config = APIConfig()

# Model loader
model_loader = ModelLoader(model_config)

@app.on_event("startup")
async def startup_event():
    """Charger modèles au démarrage"""
    print("Démarrage du service...")

    # Pré-charger modèle par défaut (NF4)
    default_quant = QuantizationConfig(method="nf4", bits=4)
    model_loader.load_model(default_quant)

    print("Service prêt!")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    Générer du texte avec quantization spécifiée

    Args:
        request: Paramètres de génération

    Returns:
        response: Texte généré et métriques
    """

    try:
        # Charger modèle
        quant_config = QuantizationConfig(
            method=request.quantization,
            bits=4  # Peut être paramétré
        )
        model = model_loader.load_model(quant_config)
        tokenizer = model_loader.get_tokenizer()

        # Tokenize
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

        # Génération
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Métriques
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None

        return GenerationResponse(
            generated_text=generated_text,
            quantization_method=request.quantization,
            num_tokens=num_tokens,
            latency_ms=latency_ms,
            memory_gb=memory_gb
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark/{quantization}", response_model=BenchmarkResponse)
async def benchmark_quantization(quantization: str):
    """
    Benchmarker une méthode de quantization

    Args:
        quantization: Méthode (fp16, int8, nf4, gptq, awq)

    Returns:
        metrics: Métriques de performance
    """

    try:
        # Charger modèle
        quant_config = QuantizationConfig(method=quantization, bits=4)
        model = model_loader.load_model(quant_config)
        tokenizer = model_loader.get_tokenizer()

        # Test prompt
        test_prompt = "Explain artificial intelligence:"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        # Warmup
        for _ in range(3):
            _ = model.generate(**inputs, max_new_tokens=10)

        # Benchmark
        num_runs = 10
        latencies = []
        num_tokens = 50

        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append(time.time() - start)

        # Métriques
        avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
        throughput = num_tokens / (avg_latency_ms / 1000)
        memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        # Compression ratio (vs FP16 baseline ~14GB pour Llama 2 7B)
        baseline_memory = 13.5
        compression_ratio = baseline_memory / memory_gb if memory_gb > 0 else 0.0

        return BenchmarkResponse(
            quantization_method=quantization,
            memory_gb=memory_gb,
            latency_ms=avg_latency_ms,
            throughput_tokens_per_sec=throughput,
            compression_ratio=compression_ratio
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

@app.post("/clear-cache")
async def clear_model_cache():
    """Vider le cache des modèles"""
    model_loader.clear_cache()
    return {"status": "cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=api_config.host,
        port=api_config.port,
        log_level="info"
    )
```

### 11.4 Utilisation du Service

**Démarrer le serveur** :
```bash
# Installation dépendances
pip install -r requirements.txt

# Lancer serveur
python api/app.py
```

**Requêtes API** :

```python
import requests

# 1. Génération avec NF4 (défaut)
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Explain quantization in simple terms:",
    "quantization": "nf4",
    "max_tokens": 100,
    "temperature": 0.7
})

print(response.json())
# {
#     "generated_text": "Explain quantization in simple terms: Quantization...",
#     "quantization_method": "nf4",
#     "num_tokens": 87,
#     "latency_ms": 1245.67,
#     "memory_gb": 3.82
# }

# 2. Benchmark INT8
response = requests.get("http://localhost:8000/benchmark/int8")

print(response.json())
# {
#     "quantization_method": "int8",
#     "memory_gb": 6.99,
#     "latency_ms": 387.23,
#     "throughput_tokens_per_sec": 129.13,
#     "compression_ratio": 1.93
# }

# 3. Health check
response = requests.get("http://localhost:8000/health")
print(response.json())
# {"status": "healthy", "gpu_available": true}
```

**Test avec cURL** :
```bash
# Génération
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "quantization": "int8",
    "max_tokens": 50
  }'

# Benchmark
curl "http://localhost:8000/benchmark/nf4"
```

---

## 12. Best Practices

### 12.1 Choisir la Bonne Quantization

**Arbre de décision** :

```
┌─────────────────────────────────────────────────────────────┐
│              Choisir la Quantization Optimale                │
└─────────────────────────────────────────────────────────────┘

1. Hardware disponible?
   ├─ GPU disponible
   │  ├─ Mémoire GPU > 40GB → FP16 (qualité maximale)
   │  ├─ Mémoire GPU 16-40GB → INT8 (BitsAndBytes)
   │  └─ Mémoire GPU < 16GB → NF4/GPTQ/AWQ 4-bit
   │
   └─ CPU uniquement
      └─ GGUF (llama.cpp) Q4_K_M ou Q5_K_M

2. Use case?
   ├─ Production inference (latence critique)
   │  └─ AWQ 4-bit (meilleure latence)
   │
   ├─ Fine-tuning efficient
   │  └─ QLoRA (NF4 + LoRA)
   │
   ├─ Compression maximale
   │  └─ GPTQ INT3/INT2
   │
   └─ Edge/mobile deployment
      └─ GGUF Q4_0 ou Q3_K_M

3. Contrainte qualité?
   ├─ Perte < 1% acceptable → INT8 ou 4-bit
   ├─ Perte 1-3% acceptable → INT4 (toutes méthodes)
   └─ Perte 3-5% acceptable → INT3 (GPTQ)
```

### 12.2 Recommandations par Modèle

| Taille Modèle | GPU Memory | Quantization Recommandée | Compression | Qualité |
|---------------|------------|--------------------------|-------------|---------|
| < 3B params   | 8GB        | INT8 ou NF4              | 2-4×        | 99%     |
| 7B params     | 16GB       | NF4 ou AWQ 4-bit         | 3-4×        | 95-98%  |
| 13B params    | 24GB       | NF4 4-bit                | 3-4×        | 95-97%  |
| 30B params    | 40GB       | NF4 4-bit                | 3-4×        | 94-96%  |
| 70B params    | 80GB       | GPTQ/AWQ 4-bit           | 4×          | 93-95%  |
| 70B params    | 48GB       | GPTQ 3-bit               | 6×          | 88-92%  |

### 12.3 Guidelines de Déploiement

**1. Calibration Data**
- Utiliser données **représentatives** du use case
- Minimum 128 exemples, optimal 512-1024
- Couvrir diversité des requêtes

**2. Validation Post-Quantization**
- Toujours benchmarker perplexity ou task-specific metrics
- Comparer outputs FP16 vs quantized sur exemples critiques
- Valider génération (pas seulement latence)

**3. Optimisations Additionnelles**
- **Fuse layers** : Conv+BN+ReLU
- **KV cache quantization** : Quantizer aussi le cache attention
- **Per-channel quantization** : Meilleure précision

**4. Monitoring Production**
```python
# Tracker métriques quantization en production
metrics = {
    'model_memory_gb': torch.cuda.memory_allocated() / 1e9,
    'quantization_method': 'nf4',
    'average_latency_ms': 245.3,
    'throughput_tokens_per_sec': 187.2,
    'quality_metric': 0.95  # Perplexity ou accuracy
}

# Logger ou envoyer à monitoring system
logger.info(f"Quantization metrics: {metrics}")
```

### 12.4 Troubleshooting Commun

**Problème 1** : Accuracy drop important après quantization

**Solutions** :
- Utiliser **per-channel quantization** au lieu de per-tensor
- Augmenter nombre d'exemples calibration
- Essayer **QAT** (Quantization-Aware Training)
- Pour outliers: **LLM.int8()** avec mixed-precision

**Problème 2** : Latence plus élevée qu'attendue

**Solutions** :
- Vérifier **fused operators** activés
- Utiliser **AWQ** au lieu de GPTQ (plus rapide)
- Sur CPU: **GGUF avec BLAS** optimisé (OpenBLAS, MKL)
- Offload layers sur GPU si disponible

**Problème 3** : Out-of-memory pendant quantization

**Solutions** :
- Quantizer **layer-by-layer** au lieu de tout le modèle
- Utiliser **gradient checkpointing**
- Réduire batch size calibration
- Pour GPTQ: réduire `group_size`

### 12.5 Checklist Pré-Déploiement

Avant de déployer un modèle quantizé en production :

- [ ] Benchmarker perplexity sur dataset validation
- [ ] Comparer générations qualitatives (FP16 vs quantized)
- [ ] Mesurer latence p50, p95, p99 (pas seulement moyenne)
- [ ] Vérifier memory footprint sous charge
- [ ] Tester edge cases (prompts longs, caractères spéciaux)
- [ ] Documenter méthode quantization et hyperparamètres
- [ ] Setup monitoring métriques quantization
- [ ] Préparer rollback vers FP16 si nécessaire

---

## CONCLUSION DU CHAPITRE

La **quantization** est une technique essentielle pour rendre les LLMs accessibles et déployables à grande échelle. Ce chapitre a couvert :

**Techniques fondamentales** :
- Quantization symétrique et asymétrique
- Per-tensor vs per-channel
- Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)

**Méthodes avancées** :
- **GPTQ** : Hessienne inverse pour précision optimale
- **AWQ** : Activation-aware scaling pour inference rapide
- **GGUF/llama.cpp** : Inference CPU haute performance
- **BitsAndBytes** : INT8/NF4 avec support CUDA

**Points clés à retenir** :

1. **INT8** offre 2× compression avec <1% perte qualité
2. **INT4** offre 4× compression avec 2-5% perte selon méthode
3. **AWQ** est optimal pour production (latence)
4. **GPTQ** est optimal pour compression maximale
5. **QLoRA (NF4)** est optimal pour fine-tuning efficient
6. **GGUF** est optimal pour déploiement CPU/edge

**Impact économique** :
- **Llama 2 70B** : 140GB (FP16) → 35GB (INT4) = **4× moins de GPUs**
- **Coût inference** : Réduction 60-75% avec quantization
- **Démocratisation** : Modèles SOTA accessibles sur hardware consumer

La quantization n'est plus optionnelle - c'est une **technique indispensable** pour tout déploiement LLM en 2026.

---

**Prochaines lectures** :
- **Chapitre 17** : Model Compression (Pruning, Distillation)
- **Chapitre 18** : Serving & Deployment (vLLM, TensorRT-LLM)
- **Chapitre 13** : LoRA & QLoRA (fine-tuning efficient)

---

*Fin du Chapitre 16 : Quantization*
