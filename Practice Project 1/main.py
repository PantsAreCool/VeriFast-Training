"""
TO RUN:

python main.py --mode cli --work-dir sample_logs --logs app.log server.log
python main.py --mode config --config config.json

"""

import os
import json
from pathlib import Path
from src.cli import parse_arguments
from src.log_parser import LogParser
from src.pattern_analyzer import PatternAnalyzer
from src.knowledge_base import KnowledgeBase 
from src.llm_interface import get_llm_client
from src.tools import create_log_agent_tools
from src.agent import LogAgent


def main():
    args, parser = parse_arguments()

    if args.mode == "cli":
        from src.cli import run_interactive_cli
        run_interactive_cli(args)
        
    elif args.mode == "config":
        from src.cli import run_config_mode
        run_config_mode(args, parser)

if __name__ == "__main__":
    main()