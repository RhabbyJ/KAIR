# SwinIR-ELSA Architecture Change Guide

## Problem

You trained a model with reduced parameters due to memory constraints:

- **Old config**: embed_dim=96, num_heads=[3,3,3,3,3,3], depths=[2,2,2,2,2,2]
- **New config**: embed_dim=180, num_heads=[6,6,6,6,6,6], depths=[6,6,6,6,6,6]

When trying to resume training with the larger architecture, there's a size mismatch between the saved checkpoint and the new model structure.

## Solution Applied

I've implemented **two flexible options** for you:

### Option 1: Flexible Loading (Current Configuration)

**Status**: ✅ **ACTIVE** - Currently configured

The model will:

1. Load all parameters that match between the old and new architecture
2. Initialize mismatched/missing parameters randomly
3. Continue training from iteration 995000

**Configuration**:

- `"G_param_strict": false` in `options/train_options.json`
- `"ignore_resume_network": false` in `options/train_options.json`

**When to use**: If you want to leverage the learned features from your previous training, even though many parameters won't transfer due to size differences.

### Option 2: Start Completely Fresh

**Status**: ⏸️ Available but not active

Train from scratch with the new larger architecture, ignoring all previous checkpoints.

**To enable**: In `options/train_options.json`, change:

```json
"ignore_resume_network": true
```

**When to use**: If you want a clean start with the larger architecture without any influence from the smaller model.

## What Was Changed

### 1. Enhanced `models/model_base.py`

- Improved `load_network()` function to intelligently handle size mismatches
- When `strict=False`, it now:
  - Loads matching parameters
  - Skips mismatched parameters (logs them)
  - Reports what was loaded vs initialized

### 2. Updated `main_train_psnr.py`

- Added support for `ignore_resume_network` flag
- Allows starting fresh training while keeping old checkpoints

### 3. Modified `options/train_options.json`

- Set `G_param_strict: false` to enable flexible loading
- Set `E_param_strict: false` for the EMA model
- Added `ignore_resume_network: false` option

## Current Model Architecture

Based on your `train_options.json`:

- **Task**: swinir_denoising_gray_15
- **Image size**: 128x128
- **Window size**: 8
- **Depths**: [6, 6, 6, 6, 6, 6] (6 blocks per layer, 6 layers)
- **Embed dimension**: 180
- **Num heads**: [6, 6, 6, 6, 6, 6]
- **ELSA enabled**: Yes (kernel_size=7)

## Expected Behavior

When you run training with **Option 1** (flexible loading):

```
Loading model for G [denoising/swinir_denoising_gray_15/models/995000_G.pth] ...
Size mismatch for conv_first.weight: checkpoint torch.Size([96, 1, 3, 3]) vs model torch.Size([180, 1, 3, 3]), using random initialization
Size mismatch for conv_first.bias: checkpoint torch.Size([96]) vs model torch.Size([180]), using random initialization
...
Keys in model but not in checkpoint (360):
  - layers.0.residual_group.blocks.2.norm1.weight
  - layers.0.residual_group.blocks.2.norm1.bias
  ...

Loaded XXX/YYY parameters from checkpoint
Mismatched: ZZZ, Missing: 360
```

Most parameters will be randomly initialized since the architecture changed significantly.

## Recommendations

Given the significant architectural changes (almost doubling the model size):

1. **I recommend Option 2** (start fresh) because:

   - The parameter sizes don't match, so transfer learning benefit is minimal
   - The model capacity is very different (2x embed_dim, 3x blocks per layer, 2x heads)
   - Starting fresh ensures all parameters are properly initialized for the new architecture

2. **If you choose Option 1**:

   - Expect slower initial convergence as most parameters are random
   - Monitor training carefully - if loss doesn't decrease in first few thousand iterations, switch to Option 2

3. **To preserve your old training**:
   - Your old checkpoints are safe in `denoising/swinir_denoising_gray_15/models/`
   - New checkpoints will be saved in the same directory
   - Consider backing up `denoising/swinir_denoising_gray_15/` before proceeding

## How to Proceed

### To use Option 1 (Flexible Loading - Current):

```bash
python main_train_psnr.py
```

### To use Option 2 (Start Fresh):

1. Edit `options/train_options.json`:
   ```json
   "ignore_resume_network": true
   ```
2. Run training:
   ```bash
   python main_train_psnr.py
   ```

## Additional Notes

- **Memory**: With embed_dim=180 and 6 blocks per layer, you'll need significantly more VRAM
- **Batch size**: Currently set to 1 - appropriate for large models
- **Checkpoints**: Saved every 5000 iterations (unchanged)
- **Training data**: Uses trainH directory with patches of size 128x128

---

**Status**: ✅ Ready to train
**Next step**: Run `python main_train_psnr.py` with your chosen option
