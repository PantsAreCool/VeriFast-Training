# Task 2: Design a Prompt Template Library
# Build a reusable prompt template system as a Python module. Include at least 5 templates for common business tasks: 
# email drafting, meeting summary, code review, data analysis interpretation, and customer response. 
# Each template should support variable substitution, have default values for optional parameters, and include input validation. 
# Write unit tests that verify template rendering with various inputs including edge cases (empty strings, very long inputs, special characters).


import string
from typing import Dict, Any, List, Set, Optional

class PromptTemplate:
    def __init__(
        self,
        name: str,
        template_str: str,
        required_variables: List[str],
        optional_defaults: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None
    ):
        self.name: str = name
        self.template_str: str = template_str
        self.required_variables: Set[str] = set(required_variables)
        self.optional_defaults: Dict[str, Any] = optional_defaults or {}
        self.system_message: Optional[str] = system_message

        self._all_expected: Set[str] = self.required_variables.union(self.optional_defaults.keys())
        self._validate_template_syntax()

    def _validate_template_syntax(self) -> None:
        formatter = string.Formatter()
        try:
            parsed = list(formatter.parse(self.template_str))
            found_vars = {field_name for _, field_name, _, _ in parsed if field_name is not None}
        except ValueError as e:
            raise ValueError(f"Invalid template syntax in template '{self.name}': {e}")

    def _sanitize_input(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def render(self, **kwargs) -> Dict[str, str]:
        merged_args = self.optional_defaults.copy()
        merged_args.update(kwargs)

        sanitized_args = {}
        for var in self._all_expected:
            val = merged_args.get(var)
            sanitized_val = self._sanitize_input(val)            
            sanitized_args[var] = sanitized_val


        user_message = self.template_str.format(**sanitized_args)

        payload = {"user": user_message}
        if self.system_message:
            payload["system"] = self.system_message
            
        return payload
    



# 1. Email Drafting Template
email_template = PromptTemplate(
    name="email_drafting",
    system_message="You are an expert corporate communications manager. Craft highly professional, accurate, and context-aware emails.",
    required_variables=["recipient", "purpose", "key_points"],
    optional_defaults={"tone": "professional and collaborative", "call_to_action": "Please let me know if you have any questions."},
    template_str="""Draft a business email to: {recipient}
Purpose of email: {purpose}
Tone: {tone}

Key points to include:
{key_points}

Call to action: {call_to_action}

Please output the email cleanly with a structured Subject Line and body layout."""
)



# 2. Meeting Summary Template
meeting_summary_template = PromptTemplate(
    name="meeting_summary",
    system_message="You are an executive assistant specializing in project tracking. Distill meetings into crisp, clear, and actionable summaries.",
    required_variables=["raw_transcript"],
    optional_defaults={"depth": "detailed", "project_context": "General Business Operations"},
    template_str="""Analyze the following meeting transcript within the context of: {project_context}
Generate a {depth} summary.

Transcript:
---
{raw_transcript}
---

Format the output strictly as follows:
## Executive Summary
[High-level overview]

## Key Decisions Made
- [Decision 1]

## Action Items
- [ ] [Owner] - [Action item description]"""
)



# 3. Code Review Template
code_review_template = PromptTemplate(
    name="code_review",
    system_message="You are a principal software engineer and security auditor. Provide rigorous, polite, and deeply technical code reviews.",
    required_variables=["language", "code"],
    optional_defaults={"framework": "Vanilla / Standard Library", "strictness": "high"},
    template_str="""Perform a comprehensive code review on the following {language} code.
Framework/Context: {framework}
Strictness Level: {strictness}

Code to analyze: {code}

Evaluate across these four categories and provide explicit line-item references:
1. Bugs & Logic Errors
2. Security Vulnerabilities (OWASP Top 10)
3. Performance Bottlenecks
4. Style and Idiomatic Conventions"""
)



# 4. Data Analysis Interpretation Template
data_analysis_template = PromptTemplate(
    name="data_analysis_interpretation",
    system_message="You are a lead data scientist and business intelligence expert. Extract actionable corporate narratives from raw metrics.",
    required_variables=["metrics_summary", "business_goal"],
    optional_defaults={"anomaly_check": "enabled"},
    template_str="""Interpret this dataset summary to help achieve the following goal: {business_goal}
Anomaly Check Status: {anomaly_check}

Dataset Metrics:
{metrics_summary}

Provide a structured narrative addressing:
- Primary statistical trends and insights
- Correlation with the stated business goal
- Outliers or anomalies detected (if any)
- Strategic data-driven recommendations"""
)



# 5. Customer Response Template
customer_response_template = PromptTemplate(
    name="customer_response",
    system_message="You are a senior customer success manager. Resolve complaints empathetically while protecting organization guidelines.",
    required_variables=["customer_issue", "company_policy"],
    optional_defaults={"urgency": "standard", "max_words": "200"},
    template_str="""Draft a customer support response to the following issue.
Urgency: {urgency}
Max Word Count Constraint: {max_words} words

Customer Issue:
\"\"\"{customer_issue}\"\"\"

Company Resolution Policy Matrix:
\"\"\"{company_policy}\"\"\"

Ensure the response validates the user's frustration, directly addresses their concern under company policies, and provides a clear path forward without corporate jargon."""
)


template_library = {
    "email_drafting": email_template,
    "meeting_summary": meeting_summary_template,
    "code_review": code_review_template,
    "data_analysis_interpretation": data_analysis_template,
    "customer_response": customer_response_template
}