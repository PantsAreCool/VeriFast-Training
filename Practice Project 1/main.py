"""
TO RUN:

python main.py --mode cli --work-dir sample_logs --logs app.log server.log
python main.py --mode config --config config.json

"""


import json
from pathlib import Path
from src.cli import parse_arguments
from src.log_parser import LogParser
from src.pattern_analyzer import PatternAnalyzer # New import

def main():
    args, parser = parse_arguments()
    
    if args.mode == "config":
        if not args.config:
            parser.error("--config is required when execution --mode is 'config'")
        log_parser = LogParser.from_config(args.config)
    elif args.mode == "cli":
        if not args.logs:
            parser.error("--logs list is required when execution --mode is 'cli'")
        log_parser = LogParser(work_dir=args.work_dir, logs=args.logs)

    log_parser.run()

    print("="*60)
    print("Running Pattern Analysis Pipeline...")
    print("="*60)
    
    analyzer = PatternAnalyzer(window_minutes=5)
    
    for log_file in log_parser.logs:
        log_name = Path(log_file).stem
        parsed_json_path = Path("parsed_logs") / f"{log_name}_parsed.json"
        
        if parsed_json_path.exists():
            with open(parsed_json_path, 'r') as f:
                parsed_data = json.load(f)
            
            analysis_results = analyzer.analyze(parsed_data)
            
            output_path = Path("parsed_logs") / f"{log_name}_analysis.json"
            with open(output_path, 'w') as out_f:
                json.dump(analysis_results, out_f, indent=2)
                
            print(f"[{log_file}] Analysis complete. Recommendations generated: {len(analysis_results['actionable_recommendations'])}")
            print(f"Report saved to: {output_path}")
            for rec in analysis_results["actionable_recommendations"]:
                print(f" -> {rec}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()