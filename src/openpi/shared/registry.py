"""
Generic registry module

This module provides a generic registry for managing and referencing config classes.
Mainly used for model config registration and management.
"""

import dataclasses
from typing import Any, Dict, Type, TypeVar, Generic, Optional

# Type variable for generics
T = TypeVar('T')

class Registry(Generic[T]):
    """Generic registry class"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
    
    def register(self, name: str):
        """Registration decorator"""
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                # raise ValueError(f"{self.name} '{name}' is already registered, you can ignore this error if you want to override the existing registration")
                pass
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type[T]:
        """Get registered class by name"""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"{self.name} '{name}' not found. Available: {available}")
        return self._registry[name]
    
    def list(self) -> list[str]:
        """List all registered names"""
        return list(self._registry.keys())
    
    def create(self, name: str, **kwargs) -> T:
        """Create instance from name and kwargs"""
        cls = self.get(name)
        return cls(**kwargs)
    
    def exists(self, name: str) -> bool:
        """Check whether a name is registered"""
        return name in self._registry
    
    def clear(self):
        """Clear registry"""
        self._registry.clear()


# Create model config registry instance
model_config_registry = Registry("ModelConfig")

# Helper functions
def register_model_config(name: str):
    """Decorator to register model config"""
    return model_config_registry.register(name)

def get_model_config(name: str) -> Type:
    """Get model config class by name"""
    return model_config_registry.get(name)

def list_model_configs() -> list[str]:
    """List all available model configs"""
    return model_config_registry.list()

def create_model_config(name: str, **kwargs) -> Any:
    """Create model config instance by name and kwargs"""
    return model_config_registry.create(name, **kwargs)

def model_config_exists(name: str) -> bool:
    """Check whether model config exists"""
    return model_config_registry.exists(name)

def clear_model_configs():
    """Clear model config registry"""
    model_config_registry.clear()


# Create other registry types if needed
data_config_registry = Registry("DataConfig")
optimizer_registry = Registry("Optimizer")
scheduler_registry = Registry("Scheduler")
weight_loader_registry = Registry("WeightLoader")

# Helper functions
def register_data_config(name: str):
    return data_config_registry.register(name)

def get_data_config(name: str) -> Type:
    return data_config_registry.get(name)

def list_data_configs() -> list[str]:
    return data_config_registry.list()

def create_data_config(name: str, **kwargs) -> Any:
    return data_config_registry.create(name, **kwargs)

def data_config_exists(name: str) -> bool:
    return data_config_registry.exists(name)

def clear_data_configs():
    """Clear data config registry"""
    data_config_registry.clear()

def register_optimizer_config(name: str):
    return optimizer_registry.register(name)

def get_optimizer_config(name: str) -> Type:
    return optimizer_registry.get(name)

def list_optimizer_config() -> list[str]:
    return optimizer_registry.list()

def create_optimizer_config(name: str, **kwargs) -> Any:
    return optimizer_registry.create(name, **kwargs)

def register_scheduler(name: str):
    return scheduler_registry.register(name)

def get_scheduler(name: str) -> Type:
    return scheduler_registry.get(name)

def list_schedulers() -> list[str]:
    return scheduler_registry.list()

def create_scheduler(name: str, **kwargs) -> Any:
    return scheduler_registry.create(name, **kwargs)

def register_weight_loader(name: str):
    return weight_loader_registry.register(name)

def get_weight_loader(name: str) -> Type:
    return weight_loader_registry.get(name)

def list_weight_loaders() -> list[str]:
    return weight_loader_registry.list()

def create_weight_loader(name: str, **kwargs) -> Any:
    return weight_loader_registry.create(name, **kwargs)


def weight_loader_exists(name: str) -> bool:
    return weight_loader_registry.exists(name)


def clear_weight_loaders():
    weight_loader_registry.clear()




# Generic registration function
def register_all():
    """Register all available configs"""
    # Auto-registration logic can be added here
    pass


# Export all registry instances for advanced usage
__all__ = [
    'Registry',
    'model_config_registry',
    'data_config_registry', 
    'optimizer_registry',
    'scheduler_registry',
    'register_model_config',
    'get_model_config',
    'list_model_configs',
    'create_model_config',
    'model_config_exists',
    'clear_model_configs',
    'register_data_config',
    'get_data_config',
    'list_data_configs',
    'create_data_config',
    'data_config_exists',
    'clear_data_configs',
    'register_optimizer',
    'get_optimizer',
    'list_optimizers',
    'create_optimizer',
    'register_scheduler',
    'get_scheduler',
    'list_schedulers',
    'create_scheduler',
] 