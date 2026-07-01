import argparse
from typing import List, Tuple

def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Configures and runs argument parsing for log processing styles"""
    parser = argparse.ArgumentParser(description="Parse log files and generate informative JSON reports")
    
    parser.add_argument("--mode", choices=["cli", "config"], default="config", 
                        help="Choose 'config' to read configuration file or 'cli' to pass target items directly.")
    
    parser.add_argument("--config", help="Path to JSON config file containing work directory and log file list")
    
    parser.add_argument("--logs", nargs="+", help="Space-separated list of log files or directory paths to analyze directly")
    parser.add_argument("--work-dir", default=".", help="Working directory to find logs passed manually via --logs")

    args = parser.parse_args()
    return args, parser