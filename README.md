# ZipMamba

An efficient ASR (Automatic Speech Recognition) encoder combining U-Net hierarchical downsampling with Bidirectional Mamba (State Space Models).

## Overview

ZipMamba aims to achieve competitive transcription quality while significantly reducing computational cost compared to attention-based models like Conformer. The architecture replaces O(n^2) self-attention with O(n) Mamba SSM blocks while preserving Conformer-style convolution modules for local feature extraction.

### Key Features

- **U-Net encoder structure**: Progressive downsampling (50Hz to 25Hz to 12.5Hz to 6.25Hz) and upsampling (back to 25Hz output) with skip connections
- **Bidirectional Mamba**: O(n) linear complexity replacing quadratic self-attention
- **ConMamba blocks**: Macaron-style architecture (FFN, BiMamba, ConvModule, FFN)
- **Zipformer components**: BiasNorm, SwooshR/SwooshL activations for improved training stability
- **CTC decoding** with optional intermediate supervision

## Architecture

```
Audio (16kHz) --> Mel Spectrogram (80-dim, 100Hz)
                             |
                             v
+----------------------------------------------+
|              ConvEmbed (2D Conv)             |
|           100Hz --> 50Hz, project to dim     |
+----------------------------------------------+
                             |
                             v
+----------------------------------------------+
|            U-Net Encoder Stacks              |
|                                              |
|  Stack 1: 50Hz    --+                        |
|  Stack 2: 25Hz    --+-- Skip connections     |
|  Stack 3: 12.5Hz  --+                        |
|  Stack 4: 6.25Hz (bottleneck)                |
|  Stack 5: 12.5Hz  <-+                        |
|  Stack 6: 25Hz    <-+                        |
+----------------------------------------------+
                             |
                             v
                   CTC Projection --> Logits
```

### ConMamba Block

Each encoder stack contains multiple ConMamba blocks with Macaron-style residual connections:

```
Input
  +-- + 0.5 * FFN (SwiGLU + SwooshL)
  +-- + BiMamba (forward + backward SSM, averaged)
  +-- + ConvModule (depthwise separable conv + SwooshR)
  +-- + 0.5 * FFN
  +-- BiasNorm --> Output
```

### Components

| Component | Description |
|-----------|-------------|
| **BiMamba** | Bidirectional Mamba SSM with O(n) complexity |
| **ConvModule** | Pointwise, GLU, Depthwise Conv, BiasNorm, SwooshR, Pointwise |
| **FeedForward** | SwiGLU-style: SwooshL(W1*x) * W2*x then W3 |
| **BiasNorm** | RMS normalization with learnable bias and exp scaling |
| **Downsample** | Learnable weighted average over consecutive frames |
| **Upsample** | Frame repetition (nearest neighbor) |
| **Bypass** | Learnable alpha interpolation between stack input and output |

## Installation

Requires Python 3.12 and a CUDA GPU.

```bash
uv sync

# Install mamba-ssm (CUDA kernels required for training)
# Check your PyTorch version first:
uv run python -c "import torch; print(torch.__version__)"

# Then install the matching wheel from:
# https://github.com/state-spaces/mamba/releases
# Example for PyTorch 2.7 + CUDA 12 + Python 3.12:
uv pip install "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

> **Note**: The `mamba-ssm` package is required. The model will not work without it.

## Usage

### Training

```bash
uv run scripts/train.py \
    --config configs/model/small.yaml \
    --config configs/training/default.yaml \
    --config configs/data/common_voice_fr.yaml
```

### Evaluation

```bash
uv run scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --config configs/model/small.yaml \
    --config configs/data/common_voice_fr.yaml
```

### Transcription

```bash
uv run scripts/transcribe.py \
    --checkpoint checkpoints/best.pt \
    --audio path/to/audio.wav
```

## Model Configurations

| Model | Parameters | Stack Dimensions |
|-------|------------|------------------|
| `small.yaml` | ~35M | 128, 192, 256, 320, 256, 192 |
| `medium.yaml` | ~92M | 192, 256, 384, 512, 384, 256 |

### Small Model Configuration

```yaml
stacks:
  - {dim: 128, blocks: 2, downsample: 2}  # 50 --> 25 Hz
  - {dim: 192, blocks: 3, downsample: 2}  # 25 --> 12.5 Hz
  - {dim: 256, blocks: 2, downsample: 2}  # 12.5 --> 6.25 Hz
  - {dim: 320, blocks: 2}                 # bottleneck
  - {dim: 256, blocks: 2, upsample: 2}    # 6.25 --> 12.5 Hz
  - {dim: 192, blocks: 3, upsample: 2}    # 12.5 --> 25 Hz
```

## Performance

> Results on Common Voice (test split), greedy CTC decoding.
> "Normalized" metrics ignore punctuation and special characters (only letters, digits and spaces are compared).

TODO: beam search + LM decoding, EN training.

### French (Common Voice FR)

| Model | WER (%) | CER (%) | WER norm. (%) | CER norm. (%) | RTF |
|-------|---------|---------|---------------|---------------|-----|
| Small (~35M) | 16.67 | 6.59 | 14.35 | 6.21 | 0.0002 |
| Medium (~92M) | 15.20 | 5.83 | 12.87 | 5.45 | 0.0003 |

#### Breakdown by validation quality

> Common Voice clips are community-validated: `down_votes` counts reviewers who flagged the audio as not matching the text. ~19% of the test set has `down_votes > 0`.

##### Small (~35M)

| Subset | N | WER (%) | CER (%) | WER norm. (%) | CER norm. (%) |
|--------|------|---------|---------|---------------|---------------|
| down_votes == 0 | 13097 | 15.26 | 5.78 | 12.99 | 5.40 |
| down_votes > 0 | 3099 | 22.55 | 9.96 | 19.99 | 9.53 |

##### Medium (~92M)

| Subset | N | WER (%) | CER (%) | WER norm. (%) | CER norm. (%) |
|--------|------|---------|---------|---------------|---------------|
| down_votes == 0 | 13096 | 13.86 | 5.08 | 11.58 | 4.71 |
| down_votes > 0 | 3100 | 20.78 | 8.96 | 18.23 | 8.55 |

### English (Common Voice EN)

| Model | WER (%) | CER (%) | WER norm. (%) | CER norm. (%) | RTF |
|-------|---------|---------|---------------|---------------|-----|
| Small (~35M) | - | - | - | - | - |
| Medium (~92M) | - | - | - | - | - |

### French + English (Common Voice FR+EN)

| Model | WER (%) | CER (%) | WER norm. (%) | CER norm. (%) | RTF |
|-------|---------|---------|---------------|---------------|-----|
| Small (~35M) | - | - | - | - | - |
| Medium (~92M) | - | - | - | - | - |

### Training Progress

Training logs are available in TensorBoard:

```bash
tensorboard --logdir logs/
```

## References

This project builds upon ideas from:

- **Zipformer** - Yao et al., "Zipformer: A faster and better encoder for automatic speech recognition", ICLR 2024
  - U-Net downsampling structure, BiasNorm, SwooshR/SwooshL activations, bypass connections

- **Mamba** - Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
  - Selective State Space Model architecture used in BiMamba blocks

- **Conformer** - Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech 2020
  - Convolution module design (depthwise separable convolution with GLU gating)

## License

MIT
