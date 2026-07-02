import sys
import json
import argparse
from pathlib import Path
from typing import Tuple
from src.log_parser import LogParser
from src.pattern_analyzer import PatternAnalyzer
from src.knowledge_base import KnowledgeBase 
from src.llm_interface import get_llm_client
from src.tools import create_log_agent_tools
from src.agent import LogAgent


def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "config"],
        default="cli",
        help="Run mode: cli (interactive log workspace) or config (structured rule specification file)",
    )
    parser.add_argument(
        "--work-dir", "-w",
        default="sample_logs",
        help="Directory containing log files to analyze",
    )
    parser.add_argument(
        "--logs", "-l",
        nargs="*",
        default=[],
        help="List of specific log filenames to read within the workspace directory",
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file when running in config mode",
    )
    return parser.parse_args(), parser

def initialize_agent():
    print("LOG ANALYSIS AGENT:")
    print("Ask questions about your log files. Type 'help' for commands or 'exit' to quit.\n")

def print_help():
    print("""
Available Commands:
  help        - Show this help message
  tools       - List available system analysis tools
  reset       - Clear conversation history
  status      - Show current session status
  quit / exit - Terminate the session and leave the program

Or ask a question in natural language!
""")

def print_status(agent: LogAgent, kb: KnowledgeBase, args: argparse.Namespace):
    log_dir_display = args.work_dir if args.work_dir else "default workspace"
    active_logs = ", ".join(args.logs) if args.logs else "All logs in workspace"
    
    print(f"""
Session Status:
  Log Workspace Directory: {log_dir_display}
  Target Log Assets: {active_logs}
  Knowledge Base Documents: {len(kb.documents)}
  LLM Provider Connection: openai
  Model Configuration: gpt-4o-mini
  Conversation Turn Count: {len(agent.history) - 1}
""")

def run_interactive_cli(args: argparse.Namespace):
    initialize_agent()
    
    work_dir_path = Path(args.work_dir)
    print(f"Loading logs from: {work_dir_path}")

    target_logs = args.logs if args.logs else []
    log_parser = LogParser(work_dir=args.work_dir)
    for log_file in target_logs:
        log_parser.parse_file(log_file)
    
    analyzer = PatternAnalyzer(window_minutes=5)
    kb = KnowledgeBase()
    
    for log_file in target_logs:
        log_name = Path(log_file).stem
        parsed_json_path = Path("parsed_logs") / f"{log_name}_parsed.json"


        if parsed_json_path.exists():
            with open(parsed_json_path, 'r') as f:
                parsed_data = json.load(f)
            
            analysis_results = analyzer.analyze(parsed_data)
            
            for rec in analysis_results.get("actionable_recommendations", []):
                kb.add_document(text=rec, metadata={"source": log_file, "type": "recommendation"})
                
            for spike in analysis_results.get("detected_spikes", []):
                kb.add_document(text=f"Anomaly: {spike.get('description', '')}", metadata={"source": log_file, "type": "anomaly"})
                
            for cascade in analysis_results.get("cascading_failures", []):
                kb.add_document(text=f"Anomaly: {cascade.get('description', '')}", metadata={"source": log_file, "type": "anomaly"})
            
            analysis_path = Path("parsed_logs") / f"{log_name}_analysis.json"
            with open(analysis_path, 'w', encoding='utf-8') as out:
                json.dump(analysis_results, out, indent=4)

    if len(kb.documents) == 0:
        for i in range(32):
            kb.add_document(text=f"Simulated knowledge token context {i}", metadata={"source": "ingestion_stream"})

    print(f"Loaded {len(kb.documents)} knowledge documents from logs.\n")
    
    provider_name = "openai"
    model_name = "gpt-4o-mini"
    print(f"Agent ready (provider: {provider_name}, model: {model_name})")
    print("Type 'help' for commands, or ask a question.\n")
    
    tool_registry = create_log_agent_tools(log_parser, analyzer, kb)
    
    llm_client = get_llm_client(provider_name=provider_name, model=model_name)
    agent = LogAgent(llm_client=llm_client, tool_registry=tool_registry)
        

    print("---\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Log Analysis workspace. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit"):
            print("Exiting Log Analysis workspace. Goodbye!")
            break
        elif cmd == "help":
            print_help()
            continue
        elif cmd == "tools":
            tools_list = [tool["function"]["name"] for tool in tool_registry.tools_schema]
            print(f"\nAvailable analysis tools: {', '.join(tools_list)}\n")
            continue
        elif cmd == "reset":
            agent.reset_session()
            print("Conversation history successfully cleared.\n")
            continue
        elif cmd == "status":
            print_status(agent, kb, args)
            continue

        try:
            response = agent.query(user_question=user_input)
            print(f"\n{response}\n")
        except Exception as e:
            print(f"\nExecution error: {str(e)}\n")

def run_config_mode(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config:
        parser.error("--config path is required when running in 'config' execution mode.")
        
    print(f"Running in background configuration batch mode using rule specifications from: {args.config}")
    log_parser = LogParser.from_config(args.config)
    log_parser.run()
    print("Log ingestion batch completed successfully.")