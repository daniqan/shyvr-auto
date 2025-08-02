"""
Immutable configuration wrapper
Provides immutability protection for configuration objects after initialization
"""

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class ImmutableConfig:
    """Wrapper that makes configuration objects immutable after initialization"""
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize immutable configuration wrapper"""
        self._data = self._make_immutable(config_data)
        self._locked = True
        logger.info("Configuration locked as immutable")
    
    def _make_immutable(self, obj: Any) -> Any:
        """Recursively make configuration data immutable"""
        if isinstance(obj, dict):
            return ImmutableDict(obj)
        elif isinstance(obj, list):
            return ImmutableList(obj)
        else:
            return obj
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from immutable configuration"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return getattr(self._data, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification after locking"""
        if hasattr(self, '_locked') and self._locked and not name.startswith('_'):
            raise AttributeError("Configuration is immutable and cannot be modified")
        super().__setattr__(name, value)
    
    def __getitem__(self, key: str) -> Any:
        """Get item from immutable configuration"""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Prevent item modification"""
        raise TypeError("Configuration is immutable and cannot be modified")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self._data.get(key, default)


class ImmutableDict:
    """Immutable dictionary wrapper"""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize immutable dictionary"""
        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = ImmutableDict(value)
            elif isinstance(value, list):
                self._data[key] = ImmutableList(value)
            else:
                self._data[key] = value
        self._locked = True
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from dictionary"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'ImmutableDict' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification after locking"""
        if hasattr(self, '_locked') and self._locked and not name.startswith('_'):
            raise AttributeError("Configuration is immutable and cannot be modified")
        super().__setattr__(name, value)
    
    def __getitem__(self, key: str) -> Any:
        """Get item from immutable dictionary"""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Prevent item modification"""
        raise TypeError("Configuration is immutable and cannot be modified")
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in dictionary"""
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default"""
        return self._data.get(key, default)
    
    def keys(self):
        """Get dictionary keys"""
        return self._data.keys()
    
    def values(self):
        """Get dictionary values"""
        return self._data.values()
    
    def items(self):
        """Get dictionary items"""
        return self._data.items()


class ImmutableList:
    """Immutable list wrapper"""
    
    def __init__(self, data: List[Any]):
        """Initialize immutable list"""
        self._data = []
        for item in data:
            if isinstance(item, dict):
                self._data.append(ImmutableDict(item))
            elif isinstance(item, list):
                self._data.append(ImmutableList(item))
            else:
                self._data.append(item)
        self._locked = True
    
    def __getitem__(self, index: int) -> Any:
        """Get item from immutable list"""
        return self._data[index]
    
    def __setitem__(self, index: int, value: Any) -> None:
        """Prevent item modification"""
        raise TypeError("Configuration is immutable and cannot be modified")
    
    def __len__(self) -> int:
        """Get list length"""
        return len(self._data)
    
    def __iter__(self):
        """Iterate over list"""
        return iter(self._data)
    
    def append(self, value: Any) -> None:
        """Prevent list modification"""
        raise TypeError("Configuration is immutable and cannot be modified")
    
    def extend(self, values: List[Any]) -> None:
        """Prevent list modification"""
        raise TypeError("Configuration is immutable and cannot be modified")
    
    def remove(self, value: Any) -> None:
        """Prevent list modification"""
        raise TypeError("Configuration is immutable and cannot be modified")
    
    def pop(self, index: int = -1) -> Any:
        """Prevent list modification"""
        raise TypeError("Configuration is immutable and cannot be modified")


def make_config_immutable(obj: Any) -> Union[ImmutableConfig, ImmutableDict, ImmutableList, Any]:
    """Helper function to make any configuration object immutable"""
    if isinstance(obj, dict):
        return ImmutableDict(obj)
    elif isinstance(obj, list):
        return ImmutableList(obj)
    else:
        return obj