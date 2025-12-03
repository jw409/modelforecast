"""Output formatters for JSON and Markdown reports."""

from modelforecast.output.json_report import write_json_report
from modelforecast.output.markdown_report import write_markdown_report

__all__ = [
    "write_json_report",
    "write_markdown_report",
]
