## Some configurations
import jax

jax.config.update("jax_enable_x64", True)# Enable 64-bit once, globally

def jit (*args, **kwargs): # A simple wrapper for JAX's jit function to set the backend device
    device = 'cpu'
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)
