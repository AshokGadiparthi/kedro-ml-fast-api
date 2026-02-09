"""
Robust JSON Serialization Utils - Handles ANY data type
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal


class RobustJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder that handles:
    - NaN, Inf, -Inf → Convert to None or specific string
    - datetime, date, time → ISO format strings
    - Decimal → float
    - numpy types → Python native types
    - pandas NA → None
    """
    
    def default(self, obj):
        # Handle NaN and Inf
        if isinstance(obj, float):
            if np.isnan(obj):
                return None  # Or "NaN" if you prefer string
            elif np.isinf(obj):
                return None  # Or "Inf" if you prefer string
            return obj
        
        # Handle numpy types
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        
        # Handle numpy NaN/Inf
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle pandas NA
        elif pd.isna(obj):
            return None
        
        # Handle datetime types
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        
        # Handle Decimal
        elif isinstance(obj, Decimal):
            return float(obj)
        
        # Handle pandas types
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        
        # Fallback
        return super().default(obj)


def serialize_for_json(obj):
    """
    Recursively serialize any object to JSON-safe format
    
    Handles:
    - Dictionaries with NaN values
    - Lists with mixed types
    - Nested structures
    - All numpy/pandas types
    """
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    
    elif isinstance(obj, tuple):
        return tuple(serialize_for_json(v) for v in obj)
    
    # Handle floats (NaN, Inf)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    
    # Handle numpy types
    elif isinstance(obj, (np.integer, np.floating)):
        if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return serialize_for_json(obj.tolist())
    
    # Handle pandas NA/NaT
    elif pd.isna(obj):
        return None
    
    # Handle datetime
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    elif isinstance(obj, timedelta):
        return str(obj)
    
    # Handle Decimal
    elif isinstance(obj, Decimal):
        return float(obj)
    
    # Handle pandas types
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    
    # Return as-is for basic types
    return obj


def safe_json_dumps(obj, **kwargs):
    """
    Safe JSON serialization with custom encoder
    
    Usage:
        json_str = safe_json_dumps(data)
    """
    return json.dumps(obj, cls=RobustJSONEncoder, **kwargs)


def safe_json_loads(json_str):
    """
    Safe JSON deserialization
    """
    return json.loads(json_str)


# Example usage:
if __name__ == "__main__":
    test_data = {
        "numeric": 42,
        "float": 3.14,
        "nan": np.nan,
        "inf": np.inf,
        "list": [1, 2, np.nan, 4],
        "datetime": datetime.now(),
        "date": date.today(),
        "nested": {
            "value": np.float64(1.5),
            "missing": None,
            "nan": float('nan')
        }
    }
    
    # Serialize safely
    json_str = safe_json_dumps(test_data)
    print(json_str)
    
    # Deserialize
    data = safe_json_loads(json_str)
    print(data)
