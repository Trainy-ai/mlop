# Neptune to mlop Migration Guide

This guide helps you migrate your training code from Neptune to mlop using the dual-logging compatibility layer.

## Overview

The `mlop.compat.neptune` module provides a **zero-code-change** migration path from Neptune to mlop. During the transition period, your code will log to **both Neptune and mlop simultaneously**, ensuring:

- ✅ **No disruption** to existing Neptune workflows
- ✅ **Gradual migration** over the 2-month transition period
- ✅ **Fallback safety** - if mlop is down, Neptune continues working
- ✅ **Simple activation** - just one import line

## Quick Start (3 Steps)

### Step 1: Add the compatibility import

At the top of your existing Neptune training script, add:

```python
import mlop.compat.neptune
```

That's it! This single line enables dual-logging.

### Step 2: Configure mlop credentials

Set environment variables before running your script:

```bash
# Required
export MLOP_PROJECT="your-project-name"

# Optional (falls back to keyring if not set)
export MLOP_API_KEY="your-mlop-api-key"
```

### Step 3: Run your script

Run your existing script normally:

```bash
python your_training_script.py
```

Your metrics, configs, images, and histograms will now be logged to **both Neptune and mlop**!

## Configuration Options

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MLOP_PROJECT` | **Yes** | mlop project name | `my-team-project` |
| `MLOP_API_KEY` | No | API key (falls back to keyring) | `mlop_api_xxx...` |
| `MLOP_URL_APP` | No | Custom app URL | `https://mlop.company.com` |
| `MLOP_URL_API` | No | Custom API URL | `https://mlop-api.company.com` |
| `MLOP_URL_INGEST` | No | Custom ingest URL | `https://mlop-ingest.company.com` |

### Credential Priority

The monkeypatch tries credentials in this order:

1. **Environment variable** (`MLOP_API_KEY`)
2. **Keyring** (stored via `mlop auth`)
3. **Graceful fallback** - if both fail, continues with Neptune-only logging

## Supported Neptune Features

The compatibility layer supports the key Neptune APIs your team is using:

| Neptune API | mlop Equivalent | Status |
|-------------|-----------------|--------|
| `Run()` | `mlop.init()` | ✅ Supported |
| `log_metrics(data, step)` | `run.log(data)` | ✅ Supported |
| `log_configs(data)` | `config={}` in init | ✅ Supported |
| `assign_files(files)` | `run.log({"key": mlop.Image()})` | ✅ Supported |
| `log_files(files, step)` | `run.log({"key": mlop.Image()})` | ✅ Supported |
| `log_histograms(hists, step)` | `run.log({"key": mlop.Histogram()})` | ✅ Supported |
| `add_tags(tags)` | Stored in config | ✅ Supported |
| `close()` | `run.finish()` | ✅ Supported |

### Automatic Type Conversion

The monkeypatch automatically converts Neptune types to mlop types:

- **Images**: `neptune_scale.types.File` → `mlop.Image`
- **Audio**: `neptune_scale.types.File` → `mlop.Audio`
- **Video**: `neptune_scale.types.File` → `mlop.Video`
- **Histograms**: `neptune_scale.types.Histogram` → `mlop.Histogram`

## Example Code

### Before (Original Neptune Code)

```python
from neptune_scale import Run

run = Run(experiment_name="my-experiment")
run.log_configs({"lr": 0.001, "batch_size": 32})

for step in range(100):
    run.log_metrics({"loss": 0.5, "acc": 0.9}, step=step)

run.close()
```

### During Transition (Dual-Logging)

```python
import mlop.compat.neptune  # ← ADD THIS LINE

from neptune_scale import Run

run = Run(experiment_name="my-experiment")
run.log_configs({"lr": 0.001, "batch_size": 32})

for step in range(100):
    run.log_metrics({"loss": 0.5, "acc": 0.9}, step=step)

run.close()
```

Now logs to **both Neptune and mlop**!

### After Migration (mlop Only)

```python
import mlop

run = mlop.init(
    project="my-project",
    name="my-experiment",
    config={"lr": 0.001, "batch_size": 32}
)

for step in range(100):
    run.log({"loss": 0.5, "acc": 0.9})

run.finish()
```

## Error Handling & Guarantees

### Hard Guarantees

1. **Neptune never fails** due to mlop errors
2. **All exceptions caught** - mlop errors are logged as warnings
3. **Zero API changes** - Neptune's behavior is unchanged
4. **Silent fallback** - if mlop is down/misconfigured, continues with Neptune only

### What Happens When...

| Scenario | Behavior |
|----------|----------|
| `MLOP_PROJECT` not set | Continues with Neptune-only logging |
| mlop API key invalid | Logs warning, continues with Neptune-only |
| mlop service is down | Logs warning, continues with Neptune-only |
| mlop.init() fails | Logs warning, continues with Neptune-only |
| mlop.log() fails | Logs debug message, continues with Neptune |
| Network timeout | Caught internally by mlop, Neptune unaffected |

All errors are logged but **never raised** to ensure Neptune continues working.

## Testing Your Migration

### 1. Verify Dual-Logging Works

Run a simple test script:

```python
import mlop.compat.neptune
from neptune_scale import Run

run = Run(experiment_name="test-dual-log")
run.log_metrics({"test_metric": 1.0}, step=0)
run.close()

# Check both:
# 1. Neptune UI - should see the run
# 2. mlop UI - should also see the run
```

### 2. Verify Neptune Fallback

Test that Neptune works even if mlop fails:

```bash
# Set invalid mlop config
export MLOP_PROJECT="nonexistent-project"
export MLOP_API_KEY="invalid-key"

# Run your script - should still work with Neptune
python your_training_script.py
```

### 3. Run CI Tests

We've provided comprehensive tests in `tests/test_neptune_compat.py`:

```bash
# Run all compatibility tests
pytest tests/test_neptune_compat.py -v

# Run specific test categories
pytest tests/test_neptune_compat.py::TestNeptuneCompatBasic -v
pytest tests/test_neptune_compat.py::TestNeptuneCompatErrorHandling -v
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Training with Dual-Logging

on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          pip install neptune-scale trainy-mlop

      - name: Run training with dual-logging
        env:
          NEPTUNE_API_TOKEN: ${{ secrets.NEPTUNE_API_TOKEN }}
          MLOP_PROJECT: ${{ secrets.MLOP_PROJECT }}
          MLOP_API_KEY: ${{ secrets.MLOP_API_KEY }}
        run: |
          python train.py
```

### Docker Example

```dockerfile
FROM python:3.10

# Install dependencies
RUN pip install neptune-scale trainy-mlop

# Copy training code
COPY train.py /app/train.py
WORKDIR /app

# Set mlop config via env vars
ENV MLOP_PROJECT="my-project"

# Run training (will dual-log to Neptune and mlop)
CMD ["python", "train.py"]
```

## Migration Timeline

### Month 1: Dual-Logging Phase
- ✅ Add `import mlop.compat.neptune` to all training scripts
- ✅ Set `MLOP_PROJECT` environment variable
- ✅ Verify data appears in both Neptune and mlop
- ✅ Engineers continue using Neptune UI as primary

### Month 2: Transition Phase
- 🔄 Gradually shift to using mlop UI
- 🔄 Verify all features work in mlop
- 🔄 Train team on mlop UI and features
- ✅ Keep Neptune as backup

### After Month 2: Full Migration
- 🎯 Remove `import mlop.compat.neptune` line
- 🎯 Replace Neptune imports with mlop
- 🎯 Update API calls to use mlop directly (optional - can keep compat layer)
- 🎯 Decommission Neptune

## Troubleshooting

### Issue: mlop not receiving logs

**Check:**
1. Is `MLOP_PROJECT` set? (`echo $MLOP_PROJECT`)
2. Are credentials valid? Try `mlop auth status`
3. Check script output for warnings
4. Verify network connectivity to mlop service

**Debug:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

import mlop.compat.neptune  # Will print debug messages
```

### Issue: Neptune stopped working

**This should never happen!** If Neptune fails after adding the monkeypatch:

1. File a bug report immediately
2. Remove `import mlop.compat.neptune` as a workaround
3. Check if Neptune credentials are valid

### Issue: Different step numbers in Neptune vs mlop

**Expected behavior:**
- Neptune uses explicit `step` parameter
- mlop auto-increments steps
- Steps may not align perfectly during dual-logging
- This is acceptable during the transition period

**After migration:** Use mlop's auto-increment, or manually set steps if needed.

## FAQ

**Q: Do I need to change any existing code?**
A: No! Just add one import line: `import mlop.compat.neptune`

**Q: What if mlop is down during training?**
A: Your training continues normally and logs to Neptune only.

**Q: Can I disable dual-logging temporarily?**
A: Yes, just unset `MLOP_PROJECT`: `unset MLOP_PROJECT`

**Q: Will this slow down my training?**
A: Minimal impact - mlop uses async logging and batching.

**Q: Can I use this in production?**
A: Yes! It's designed for production use during the migration period.

**Q: What happens after the 2-month window?**
A: You can continue using the compat layer, or migrate to mlop API directly.

## Support

- **Issues**: https://github.com/your-org/mlop/issues
- **Docs**: https://docs.mlop.yourcompany.com
- **Slack**: #mlop-support

## See Also

- [mlop Documentation](https://docs.mlop.yourcompany.com)
- [Neptune Migration Examples](./neptune_migration_example.py)
- [Test Suite](../tests/test_neptune_compat.py)
