import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any
from src.error_extractor import ErrorExtractor

class LogParser:
    """Parse and analyze log files"""
    def __init__(self, work_dir: str = "sample_logs", output_dir: str = "parsed_logs"):
        self.work_dir = Path(work_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_extractor = ErrorExtractor()
        self.logs = []
        
    def parse_file(self, filename: str) -> Dict[str, Any]:
        file_path = self.work_dir / Path(filename).name
        
        if not file_path.exists():
            print(f"Error: Target log file not found at {file_path}")
            return {"total_lines": 0, "levels": {}, "errors": {}, "entries": []}

        if filename not in self.logs:
            self.logs.append(filename)    
        
        entries = []
        stats = defaultdict(int)
        error_distribution = defaultdict(int)
        
        log_pattern = re.compile(r'^\[?([\d\-\s:,]+)\]?\s+\[?([A-Z]+)\]?\s+(.*)$')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                match = log_pattern.match(line)
                if match:
                    timestamp_str, level, message = match.groups()
                    stats[level] += 1
                    
                    entry = {
                        "timestamp": timestamp_str,
                        "level": level,
                        "message": message
                    }
                    
                    if level == "ERROR":
                        extracted = self.error_extractor.extract(message)
                        entry["error_details"] = extracted
                        error_code = extracted.get("error_code", "UNKNOWN")
                        error_distribution[error_code] = error_distribution.get(error_code, 0) + 1
                        
                    entries.append(entry)
        
        report = {
            "filename": file_path.name,
            "total_lines": len(entries),
            "levels": dict(stats),
            "errors": dict(error_distribution),
            "entries": entries
        }
        
        output_file = self.output_dir / f"{file_path.stem}_parsed.json"
        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump(report, out, indent=4)
            
        return report