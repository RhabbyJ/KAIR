# Fixes Applied for Architecture Change

## Summary

Fixed the model loading issue when changing from smaller parameters (embed_dim=96, 2 blocks, 3 heads) to original SwinIR parameters (embed_dim=180, 6 blocks, 6 heads).

## Changes Made

### 1. **models/model_base.py** - Enhanced Flexible Loading

**Location**: Line 168-213

**What changed**: Completely rewrote the `load_network()` function's `strict=False` behavior to:

- Load only parameters that match in both name AND size
- Skip parameters with size mismatches (logs them clearly)
- Skip missing parameters
- Report loading statistics

**Why**: The original implementation blindly tried to zip old and new state dicts together, which failed when architectures differed significantly.

### 2. **utils/utils_option.py** - Fixed Bug in Default Parameter Setting

**Location**: Lines 151, 153

**What changed**:

```python
# Before (BUGGY):
if 'D_param_strict' not in opt['path']:
if 'E_param_strict' not in opt['path']:

# After (FIXED):
if 'D_param_strict' not in opt['train']:
if 'E_param_strict' not in opt['train']:
```

**Why**: The code was checking the wrong section of the config dict, causing defaults to always override user settings.

### 3. **main_train_psnr.py** - Added Fresh Start Option

**Location**: Lines 90-113

**What changed**: Added logic to skip checkpoint loading when `ignore_resume_network` is set:

- Checks `opt["path"].get("ignore_resume_network", False)`
- If True, sets all checkpoint paths to None and iteration to 0
- Prints a clear message about starting fresh

**Why**: Provides an easy way to start fresh training without deleting old checkpoints.

### 4. **options/train_options.json** - Updated Configuration

**What changed**:

- Line 13: Added `"ignore_resume_network": false`
- Line 85-86: Changed `"G_param_strict": false` and `"E_param_strict": false`

**Why**:

- Enables flexible loading by default
- Provides option to start fresh if needed
- Properly commented for clarity

## Current Configuration

Your model is now configured with:

- **Architecture**: Original SwinIR (embed_dim=180, 6 blocks, 6 heads)
- **Loading mode**: Flexible (loads what matches, initializes the rest)
- **Checkpoint resume**: Enabled (will load from iter 995000)
- **Fresh start option**: Available (set `ignore_resume_network: true`)

## Testing Results

Configuration validation: **PASSED** ✓

- Config loads successfully
- All parameters parsed correctly:
  - `ignore_resume_network: False`
  - `G_param_strict: False`
  - `E_param_strict: False`
  - `embed_dim: 180`
  - `num_heads: [6, 6, 6, 6, 6, 6]`
  - `depths: [6, 6, 6, 6, 6, 6]`

## Expected Behavior When You Run Training

### With Current Settings (Flexible Loading)

When you run `python main_train_psnr.py`, you'll see:

```
Loading model for G [denoising/swinir_denoising_gray_15/models/995000_G.pth] ...
Size mismatch for conv_first.weight: checkpoint torch.Size([96, 1, 3, 3]) vs model torch.Size([180, 1, 3, 3]), using random initialization
Size mismatch for conv_first.bias: checkpoint torch.Size([96]) vs model torch.Size([180]), using random initialization
...
[More size mismatch messages for ~400 parameters]
...

Keys in model but not in checkpoint (360):
  - layers.0.residual_group.blocks.2.norm1.weight
  - layers.0.residual_group.blocks.2.norm1.bias
  ...and 358 more

Loaded XX/YYY parameters from checkpoint
Mismatched: ~400, Missing: 360
```

**Training will start from iteration 995001** but most parameters will be randomly initialized due to architecture change.

### Alternative: Start Fresh

To start completely fresh (iteration 0, all parameters random):

1. Edit `options/train_options.json`:

   ```json
   "ignore_resume_network": true
   ```

2. Run training:
   ```bash
   python main_train_psnr.py
   ```

You'll see:

```
ignore_resume_network is True - starting training from scratch
```

Training will start from iteration 0 with all parameters freshly initialized.

## Recommendation

Given that **most parameters won't match** (different embed_dim, more blocks, more heads), I recommend:

**Option A (Recommended)**: Start fresh

- Set `"ignore_resume_network": true`
- Clean initialization for the new larger architecture
- No confusion from partially loaded weights

**Option B**: Use flexible loading

- Keep current settings
- Some low-level features might transfer
- Monitor if training converges normally

## Bugs Fixed

1. **Original bug in codebase**: `utils/utils_option.py` was checking `opt['path']` instead of `opt['train']` for strict loading defaults
2. **Inflexible loading**: `strict=False` loading didn't actually handle mismatches gracefully

## Files Modified

- ✓ `models/model_base.py` - Improved loading logic
- ✓ `utils/utils_option.py` - Fixed bug in default parameter setting
- ✓ `main_train_psnr.py` - Added fresh start option
- ✓ `options/train_options.json` - Updated configuration

## Ready to Train!

Your model is now ready to train with the original SwinIR parameters. Just run:

```bash
python main_train_psnr.py
```

Check the `ARCHITECTURE_CHANGE_GUIDE.md` for more details on the two training options.
