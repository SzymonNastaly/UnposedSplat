# UnposedSplat(+)

**UnposedSplat(+)** builds on top of **NoPoSplat**, extending it to support multiple input views. This README is complementary to the main NoPoSplat README.

## Running Experiments

Before running experiments, install packages from requirements_unposedsplat.txt (make sure to install PyTorch before the Gaussian Rasterizer).
You can run experiments similarly to **NoPoSplat**, following the instructions in its `README.md`.

## Configuration Options

### 1. Enable Fast3R-Inspired Backbone (Fusion Transformer)

To enable the Fast3R-inspired backbone, update your experiment config YAML as follows:

```yaml
encoder:
  backbone:
    name: fast3r
```

Sample configuration files are provided.

### 2. Enable UnposedSplat+

To activate UnposedSplat+, change the `decoder_args` in `fast3r_backbone.py`:

```python
decoder_args = "fast3r_plus"
```

This setting may be exported to a config file in future updates.

### 3. Use VGG-T Backbone

To experiment with the VGG-T backbone, switch to the corresponding Git branch:

```bash
git checkout vggt-backbone
```

Full integration of this backbone into main is still in progress.
