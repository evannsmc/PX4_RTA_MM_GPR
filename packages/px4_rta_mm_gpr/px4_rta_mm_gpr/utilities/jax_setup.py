## Some configurations (imported before anything is compiled)
import os

import jax

jax.config.update("jax_enable_x64", True)# Enable 64-bit once, globally

# Persistent compilation cache: compiled executables are written to disk and reused by later runs
# (and by the rollout worker process), so start-up skips most of the ~10 s of XLA compilation.
# Set PX4_RTA_JAX_CACHE='' to disable.
_cache_dir = os.environ.get('PX4_RTA_JAX_CACHE', os.path.expanduser('~/.cache/px4_rta_mm_gpr/jax'))
if _cache_dir:
    os.makedirs(_cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def jit(*args, **kwargs):
    """Kept for backwards compatibility; the CPU is JAX's default (and only) device here."""
    kwargs.pop('backend', None)
    return jax.jit(*args, **kwargs)
