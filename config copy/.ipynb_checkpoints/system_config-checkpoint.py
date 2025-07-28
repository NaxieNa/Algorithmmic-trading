"""
System Configuration Loader
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from .config_template import load_config as load_template_config

# Load environment variables
load_dotenv()

class SystemConfig:
    """System configuration manager"""
    
    def __init__(self, config_override: Dict = None):
        # Load base configuration
        self.config = load_template_config(config_override)
        
        # Apply environment overrides
        self._apply_env_overrides()
        
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        
        # Override from environment
        if os.getenv('INITIAL_CAPITAL'):
            self.config['trading']['initial_capital'] = float(os.getenv('INITIAL_CAPITAL'))
            
        if os.getenv('MAX_DRAWDOWN'):
            self.config['risk']['max_drawdown'] = float(os.getenv('MAX_DRAWDOWN'))
            
        if os.getenv('LOG_LEVEL'):
            self.config['base']['log_level'] = os.getenv('LOG_LEVEL')
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def to_dict(self) -> Dict:
        """Get full configuration as dictionary"""
        return self.config.copy()

# Global configuration instance
_config = None

def get_config(reload: bool = False) -> SystemConfig:
    """Get global configuration instance"""
    global _config
    
    if _config is None or reload:
        _config = SystemConfig()
        
    return _config

def reset_config():
    """Reset global configuration"""
    global _config
    _config = None