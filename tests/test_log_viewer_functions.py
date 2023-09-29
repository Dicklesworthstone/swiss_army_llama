import os
import pytest
from datetime import datetime, timedelta
from pytz import timezone
from log_viewer_functions import safe_highlight_func, highlight_rules_func, show_logs_func, show_logs_incremental_func

@pytest.fixture(scope="session", autouse=True)
def prepare_log_file():
    # Create a more realistic sample log file for testing
    log_file_path = 'swiss_army_llama.log'
    now = datetime.now(timezone('UTC'))
    five_minutes_ago = now - timedelta(minutes=5)
    sample_logs = f"""{five_minutes_ago.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - This is a success.
{now.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - This is an error.
"""
    with open(log_file_path, 'w') as f:
        f.write(sample_logs)
    yield
    os.remove(log_file_path)

def test_safe_highlight_func():
    assert safe_highlight_func("Hello world", r"world", "WORLD") == "Hello WORLD"
    assert safe_highlight_func("Hello world", r"[", "WORLD") == "Hello world"

def test_highlight_rules_func():
    assert highlight_rules_func("This is a success.") == '<span style="color: #baffc9;">This</span> is a <span style="color: #baffc9;">success</span>.'

def test_show_logs_func(prepare_log_file):
    logs = show_logs_func(5)
    assert "success" in logs
    assert "error" in logs

def test_show_logs_incremental_func(prepare_log_file):
    result = show_logs_incremental_func(5, 0)
    assert "success" in result["logs"]
    assert "error" in result["logs"]
