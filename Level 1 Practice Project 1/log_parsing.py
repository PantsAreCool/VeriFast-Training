'''
run via python log_parsing.py --config "input json file"
'''

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any


class LogParser:
    """Parse and analyze log files"""
    
    def __init__(self, config_path: str):
        """Initialize parser with config file"""
        self.config = self._load_config(config_path)
        self.work_dir = Path(self.config.get("wor_dir", "."))
        self.logs = self.config.get("logs", [])
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load JSON config file"""

        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _parse_log_line(self, line: str) -> Dict[str, Any]:
        """
        Parse a single log line
        Format: YYYY-MM-DD HH:MM:SS [LEVEL] message
        """
        pattern = r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.*)'
        match = re.match(pattern, line.strip())
        
        if match:
            return {
                "timestamp": f"{match.group(1)} {match.group(2)}",
                "level": match.group(3),
                "message": match.group(4)
            }
        return None
    
    def _extract_error_info(self, message: str) -> Dict[str, Any]:
        """Extract information from error messages"""
        error_info = {"raw": message}
        
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
        
        return error_info
    
    def parse_log_file(self, log_file: str) -> Dict[str, Any]:
        """Parse a complete log file and return structured data"""
        log_path = self.work_dir / log_file
        
        if not log_path.exists():
            print(f"File not found: {log_path}")
            return None
        
        parsed_data = {
            "filename": log_file,
            "parsed_at": datetime.now().isoformat(),
            "statistics": {
                "total_lines": 0,
                "info_count": 0,
                "warn_count": 0,
                "error_count": 0
            },
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        error_summary = defaultdict(int)

        with open(log_path, 'r') as f:
            for line in f:
                parsed_line = self._parse_log_line(line)
                
                if not parsed_line:
                    continue
                
                parsed_data["statistics"]["total_lines"] += 1
                level = parsed_line["level"]
                
                if level == "ERROR":
                    parsed_data["statistics"]["error_count"] += 1
                    error_info = self._extract_error_info(parsed_line["message"])
                    error_key = error_info.get("error_code", error_info.get("exception_type", "Unknown Error"))
                    error_summary[str(error_key)] += 1
                    
                    parsed_data["errors"].append({
                        "timestamp": parsed_line["timestamp"],
                        "message": parsed_line["message"],
                        "details": error_info
                    })
                
                elif level == "WARN":
                    parsed_data["statistics"]["warn_count"] += 1
                    parsed_data["warnings"].append({
                        "timestamp": parsed_line["timestamp"],
                        "message": parsed_line["message"]
                    })
                
                elif level == "INFO":
                    parsed_data["statistics"]["info_count"] += 1
                    parsed_data["info"].append({
                        "timestamp": parsed_line["timestamp"],
                        "message": parsed_line["message"]
                    })
        
            parsed_data["error_summary"] = dict(error_summary)        
        return parsed_data
    
    def save_parsed_data(self, parsed_data: Dict[str, Any], output_dir: str = "parsed_logs") -> str:
        """Save parsed data to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        log_name = Path(parsed_data["filename"]).stem
        output_file = output_path / f"{log_name}_parsed.json"
        
        with open(output_file, 'w') as f:
            json.dump(parsed_data, f, indent=2)
        
        return str(output_file)
    
    def run(self):
        """Execute the full parsing pipeline"""
        print("\n" + "="*60)
        print("Log Parsing...")
        print("="*60)
        print(f"Working Directory: {self.work_dir}")
        print(f"Log Files: {len(self.logs)}")
        print("-"*60)
        
        results = []
        total_errors = 0
        total_warnings = 0
        
        for log_file in self.logs:
            print(f"\nProcessing: {log_file}")
            parsed_data = self.parse_log_file(log_file)
    
            output_file = self.save_parsed_data(parsed_data)
            
            stats = parsed_data["statistics"]
            errors = stats["error_count"]
            warnings = stats["warn_count"]
            
            total_errors += errors
            total_warnings += warnings
            
            print(f"Parsed: {stats['total_lines']} lines")
            print(f"INFO: {stats['info_count']}")
            print(f"WARN: {warnings}")
            print(f"ERROR: {errors}")

            if parsed_data["error_summary"]:
                print(f"Error Types:")
                for error_type, count in parsed_data["error_summary"].items():
                    print(f"     - {error_type}: {count}")
            
            print(f"Saved: {output_file}")
            
            results.append({
                "filename": log_file,
                "output_file": output_file,
                "stats": stats
            })
        

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Files Processed: {len(results)}")
        print(f"Total Lines: {sum(r['stats']['total_lines'] for r in results)}")
        print(f"Total INFO: {sum(r['stats']['info_count'] for r in results)}")
        print(f"Total WARNINGS: {total_warnings}")
        print(f"Total ERRORS: {total_errors}")
        print("\nLog parsing completed!")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse log files and generate informative JSON reports")
    parser.add_argument("--config", required=True, help="Path to JSON config file containing work directory and log file list")
    
    args = parser.parse_args()
    
    log_parser = LogParser(args.config)
    log_parser.run()

if __name__ == "__main__":
    main()
