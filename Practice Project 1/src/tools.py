import json
from pathlib import Path
from typing import Dict, List, Any, Callable

class ToolRegistry:
    """Registry to manage agent tool schemas and map incoming LLM tool call strings back to active Python functions."""
    def __init__(self):
        self.tools_schema: List[Dict[str, Any]] = []
        self.functions_map: Dict[str, Callable] = {}

    def register_tool(self, schema: Dict[str, Any], func: Callable):
        """Registers a tool's JSON schema and associates it with an execution function."""
        tool_name = schema["function"]["name"]
        self.tools_schema.append(schema)
        self.functions_map[tool_name] = func

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Executes a registered tool function and wraps output back into a string."""
        if name not in self.functions_map:
            return f"Error: Tool '{name}' is not recognized by the registry."

        result = self.functions_map[name](**arguments)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)




def create_log_agent_tools(log_parser: Any, pattern_analyzer: Any, knowledge_base: Any) -> ToolRegistry:
    """Function to initialize and bind components to the tool registry."""
    registry = ToolRegistry()

    # Tool 1: List Log Files
    def list_log_files() -> List[str]:
        return [str(p) for p in log_parser.logs]

    # Tool 2: Get Parsed Log Summary
    def get_parsed_summary(log_file: str) -> str:
        base_name = Path(log_file).stem.replace("_parsed", "")
        target_json = Path("parsed_logs") / f"{base_name}_parsed.json"
        
        if not target_json.exists():
            return f"Could not find parsed records for {log_file}."
            
        with open(target_json, 'r') as f:
            data = json.load(f)
            
        return f"Log File: {data['filename']} | Total Parsed Lines: {data['total_lines']} | Errors: {data['levels'].get('ERROR', 0)}"

    # Tool 3: Read Specific Analysis Report
    def get_analysis_report(log_file: str) -> Dict[str, Any]:
        log_name = Path(log_file).stem
        analysis_path = Path("parsed_logs") / f"{log_name}_analysis.json"
        if not analysis_path.exists():
            return {"error": f"Analysis report for {log_file} not found."}
        with open(analysis_path, 'r') as f:
            return json.load(f)

    # Tool 4: Query Knowledge Base
    def query_knowledge_base(query: str, limit: int = 3) -> str:
        return knowledge_base.build_llm_context(query, limit=limit)

    # Tool 5: Get All Recommendations
    def get_all_recommendations() -> List[Dict[str, Any]]:
        recs = []
        for doc_id, doc in knowledge_base.documents.items():
            if doc["metadata"].get("type") == "recommendation":
                recs.append({"id": doc_id, "text": doc["text"], "source": doc["metadata"].get("source")})
        return recs

    # Tool 6: Get Active Anomalies
    def get_active_anomalies() -> List[Dict[str, Any]]:
        anomalies = []
        for doc_id, doc in knowledge_base.documents.items():
            if doc["metadata"].get("type") == "anomaly":
                anomalies.append({"id": doc_id, "text": doc["text"], "source": doc["metadata"].get("source")})
        return anomalies


    registry.register_tool({
        "type": "function",
        "function": {
            "name": "list_log_files",
            "description": "Returns a list of all active target log files currently processed in this workspace workspace.",
            "parameters": {"type": "object", "properties": {}}
        }
    }, list_log_files)

    registry.register_tool({
        "type": "function",
        "function": {
            "name": "get_parsed_summary",
            "description": "Retrieves high-level total line counts, metrics summary stats, and breakdown counts for a log file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_file": {"type": "string", "description": "Name of the log file (e.g. 'app_server.log')"}
                },
                "required": ["log_file"]
            }
        }
    }, get_parsed_summary)

    registry.register_tool({
        "type": "function",
        "function": {
            "name": "get_analysis_report",
            "description": "Retrieves full structured pattern analysis details, cascades, time-window groupings, and targeted suggestions for a given log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_file": {"type": "string", "description": "The target log file path to fetch details from."}
                },
                "required": ["log_file"]
            }
        }
    }, get_analysis_report)

    registry.register_tool({
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "Executes keyword query search inside index memory, fetching corresponding similarity ranked document context snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to match against knowledge base (e.g., 'circuit breaker missing')"},
                    "limit": {"type": "integer", "default": 3, "description": "Max documents to pull"}
                },
                "required": ["query"]
            }
        }
    }, query_knowledge_base)

    registry.register_tool({
        "type": "function",
        "function": {
            "name": "get_all_recommendations",
            "description": "Returns all actionable recommendation items currently indexed across the system knowledge base.",
            "parameters": {"type": "object", "properties": {}}
        }
    }, get_all_recommendations)

    registry.register_tool({
        "type": "function",
        "function": {
            "name": "get_active_anomalies",
            "description": "Pulls all registered anomalous log spikes, cascades, and threshold anomalies captured inside the index memory store.",
            "parameters": {"type": "object", "properties": {}}
        }
    }, get_active_anomalies)

    return registry