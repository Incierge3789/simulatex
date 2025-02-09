from typing import Dict, Any

def validate_token_data(data: Dict[str, Any]) -> bool:
    required_fields = [
        'token_symbol',
        'total_supply',
        'circulating_supply',
        'inflation_rate',
        'staking_reward'
    ]
    
    missing_fields = [
        field for field in required_fields 
        if not data.get(field)
    ]
    
    if missing_fields:
        logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
        return False
    return True
