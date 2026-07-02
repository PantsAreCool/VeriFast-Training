import re
from typing import Dict, Any

class ErrorExtractor:
    """Extracts detailed information from error log messages using regex patterns"""

    @staticmethod
    def extract(message: str) -> Dict[str, Any]:
        """Extract information from error messages"""
        error_info = {"raw": message}
        
        db_keywords = [
            r'(Deadlock detected)', 
            r'(Connection refused|Connection timeout)', 
            r'(Constraint violation|Unique constraint)',
            r'(Replication lag|Replication error)',
            r'(Disk full|Out of memory)'
        ]
        
        db_match = re.search('|'.join(db_keywords), message, re.IGNORECASE)
        if db_match:
            error_info["error_code"] = db_match.group(0).title()
        else:
            code_match = re.search(r'\b(\d{3})\b', message)
            if code_match:
                error_info["error_code"] = code_match.group(1)
        
        service_match = re.search(r'(upstream|service)=([a-z0-9\-]+)', message)
        if service_match:
            error_info["service"] = service_match.group(2)
        
        client_match = re.search(r'(client_id|user_id|id)=([A-Z0-9\-]+)', message)
        if client_match:
            error_info["entity_id"] = client_match.group(2)
        
        reason_match = re.search(r'reason=([a-z0-9_]+)', message)
        if reason_match:
            error_info["reason"] = reason_match.group(1)
        
        exception_match = re.search(r'(\w+Error|Exception):', message)
        if exception_match:
            error_info["exception_type"] = exception_match.group(1)
            if "error_code" not in error_info:
                error_info["error_code"] = exception_match.group(1)
        
        return error_info