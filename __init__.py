try:
    from .model import PRISMFuse
except ImportError:
    from model import PRISMFuse

__all__ = ["PRISMFuse"]
