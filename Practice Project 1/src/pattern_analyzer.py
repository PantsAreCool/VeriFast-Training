from datetime import datetime, timedelta
from collections import defaultdict
import statistics
from typing import Dict, List, Any

class PatternAnalyzer:
    """Analyzes structured log data for anomalies, spikes, and cascades"""
    def __init__(self, window_minutes: int = 5):
        self.window_seconds = window_minutes * 60

    def _get_window_key(self, timestamp_str: str) -> str:
        """Buckets timestamp string into time-window string"""
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        timestamp = dt.timestamp()
        bucketed_timestamp = (timestamp // self.window_seconds) * self.window_seconds
        return datetime.fromtimestamp(bucketed_timestamp).strftime("%Y-%m-%d %H:%M")

    def analyze(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performs analysis on parsed data"""
        if not parsed_data:
            return {}

        error_windows = defaultdict(int)
        warning_windows = defaultdict(int)
        service_failures = defaultdict(lambda: defaultdict(int))

        for err in parsed_data.get("errors", []):
            win = self._get_window_key(err["timestamp"])
            error_windows[win] += 1            
            service = err.get("details", {}).get("service", "unknown")
            service_failures[win][service] += 1

        for warn in parsed_data.get("warnings", []):
            win = self._get_window_key(warn["timestamp"])
            warning_windows[win] += 1

        spikes = []
        counts = list(error_windows.values())
        if len(counts) > 2:
            mean_errors = statistics.mean(counts)
            stdev_errors = statistics.stdev(counts)
            threshold = mean_errors + (2 * stdev_errors)
        else:
            threshold = 5

        for win, count in error_windows.items():
            if count > threshold:
                spikes.append({
                    "window": win,
                    "error_count": count,
                    "severity": "CRITICAL" if count > threshold * 1.5 else "WARNING",
                    "description": f"Error count ({count}) significantly exceeded baseline threshold ({threshold:.2f})"
                })

        cascades = []
        sorted_windows = sorted(error_windows.keys())
        
        for i in range(len(sorted_windows) - 1):
            current_win = sorted_windows[i]
            next_win = sorted_windows[i+1]
            
            curr_services = set(service_failures[current_win].keys()) - {"unknown"}
            next_services = set(service_failures[next_win].keys()) - {"unknown"}
            
            if curr_services and next_services and curr_services != next_services:
                cascades.append({
                    "trigger_window": current_win,
                    "consequent_window": next_win,
                    "initial_failing_services": list(curr_services),
                    "subsequent_failing_services": list(next_services),
                    "description": f"Potential cascading failure: Outage shifted from {curr_services} to {next_services}"
                })

        recommendations = []
        if spikes:
            recommendations.append("Action Required: Review autoscaling policies or rate-limiting filters during spike windows.")
        if cascades:
            recommendations.append("Action Required: Circuit breaker design pattern missing. Implement fallback tolerances between interdependent upstream services.")
        if any("401" in str(e) or "403" in str(e) for e in parsed_data.get("errors", [])):
            recommendations.append("Security Advisory: Elevated authentication/authorization errors detected. Check credential rotators or potential brute-force vectors.")

        return {
            "analyzed_at": datetime.now().isoformat(),
            "time_window_size_minutes": self.window_seconds // 60,
            "detected_spikes": spikes,
            "cascading_failures": cascades,
            "correlations": {
                "error_peak_windows": sorted(error_windows.items(), key=lambda x: x[1], reverse=True)[:3],
                "warning_peak_windows": sorted(warning_windows.items(), key=lambda x: x[1], reverse=True)[:3]
            },
            "actionable_recommendations": recommendations
        }