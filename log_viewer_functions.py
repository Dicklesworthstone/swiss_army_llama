import re
import logging
import html
from datetime import datetime, timedelta
from pytz import timezone

log_file_path = 'swiss_army_llama.log'

def safe_highlight_func(text, pattern, replacement):
    try:
        return re.sub(pattern, replacement, text)
    except Exception as e:
        logging.warning(f"Failed to apply highlight rule: {e}")
        return text


def highlight_rules_func(text):
    rules = [
        (re.compile(r"\b(success\w*)\b", re.IGNORECASE), '#COLOR1_OPEN#', '#COLOR1_CLOSE#'),
        (re.compile(r"\b(error|fail\w*)\b", re.IGNORECASE), '#COLOR2_OPEN#', '#COLOR2_CLOSE#'),
        (re.compile(r"\b(pending)\b", re.IGNORECASE), '#COLOR3_OPEN#', '#COLOR3_CLOSE#'),
        (re.compile(r"\b(response)\b", re.IGNORECASE), '#COLOR4_OPEN#', '#COLOR4_CLOSE#'),
        (re.compile(r'\"(.*?)\"', re.IGNORECASE), '#COLOR5_OPEN#', '#COLOR5_CLOSE#'),
        (re.compile(r"\'(.*?)\'", re.IGNORECASE), "#COLOR6_OPEN#", '#COLOR6_CLOSE#'),
        (re.compile(r"\`(.*?)\`", re.IGNORECASE), '#COLOR7_OPEN#', '#COLOR7_CLOSE#'),
        (re.compile(r"\b(https?://\S+)\b", re.IGNORECASE), '#COLOR8_OPEN#', '#COLOR8_CLOSE#'),
        (re.compile(r"\b(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\b", re.IGNORECASE), '#COLOR9_OPEN#', '#COLOR9_CLOSE#'),
        (re.compile(r"\b(_{100,})\b", re.IGNORECASE), '#COLOR10_OPEN#', '#COLOR10_CLOSE#'),
        (re.compile(r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\b", re.IGNORECASE), '#COLOR11_OPEN#', '#COLOR11_CLOSE#'),
        (re.compile(r"\b([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\b", re.IGNORECASE), '#COLOR12_OPEN#', '#COLOR12_CLOSE#'),
        (re.compile(r"\b([a-f0-9]{64})\b", re.IGNORECASE), '#COLOR13_OPEN#', '#COLOR13_CLOSE#')                                
    ]
    for pattern, replacement_open, replacement_close in rules:
        text = pattern.sub(f"{replacement_open}\\1{replacement_close}", text)
    text = html.escape(text)
    text = text.replace('#COLOR1_OPEN#', '<span style="color: #baffc9;">').replace('#COLOR1_CLOSE#', '</span>')
    text = text.replace('#COLOR2_OPEN#', '<span style="color: #ffb3ba;">').replace('#COLOR2_CLOSE#', '</span>')
    text = text.replace('#COLOR3_OPEN#', '<span style="color: #ffdfba;">').replace('#COLOR3_CLOSE#', '</span>')
    text = text.replace('#COLOR4_OPEN#', '<span style="color: #ffffba;">').replace('#COLOR4_CLOSE#', '</span>')
    text = text.replace('#COLOR5_OPEN#', '<span style="color: #bdc7e7;">').replace('#COLOR5_CLOSE#', '</span>')
    text = text.replace('#COLOR6_OPEN#', "<span style='color: #d5db9c;'>").replace('#COLOR6_CLOSE#', '</span>')
    text = text.replace('#COLOR7_OPEN#', '<span style="color: #a8d8ea;">').replace('#COLOR7_CLOSE#', '</span>')
    text = text.replace('#COLOR8_OPEN#', '<span style="color: #e2a8a8;">').replace('#COLOR8_CLOSE#', '</span>')
    text = text.replace('#COLOR9_OPEN#', '<span style="color: #ece2d0;">').replace('#COLOR9_CLOSE#', '</span>')
    text = text.replace('#COLOR10_OPEN#', '<span style="color: #d6e0f0;">').replace('#COLOR10_CLOSE#', '</span>')
    text = text.replace('#COLOR11_OPEN#', '<span style="color: #f2d2e2;">').replace('#COLOR11_CLOSE#', '</span>')
    text = text.replace('#COLOR12_OPEN#', '<span style="color: #d5f2ea;">').replace('#COLOR12_CLOSE#', '</span>')
    text = text.replace('#COLOR13_OPEN#', '<span style="color: #f2ebd3;">').replace('#COLOR13_CLOSE#', '</span>')
    return text

def show_logs_incremental_func(minutes: int, last_position: int):
    new_logs = []
    now = datetime.now(timezone('UTC'))  # get current time, make it timezone-aware
    with open(log_file_path, "r") as f:
        f.seek(last_position)  # seek to `last_position`
        while True:
            line = f.readline()
            if line == "":  # if EOF
                break
            if line.strip() == "":
                continue
            try:  # Try to parse the datetime
                log_datetime_str = line.split(" - ")[0]  # assuming the datetime is at the start of each line
                log_datetime = datetime.strptime(log_datetime_str, "%Y-%m-%d %H:%M:%S,%f")  # parse the datetime string to a datetime object
                log_datetime = log_datetime.replace(tzinfo=timezone('UTC'))  # set the datetime object timezone to UTC to match `now`
                if now - log_datetime > timedelta(minutes=minutes):  # if the log is older than `minutes` minutes from now
                    continue  # ignore the log and continue with the next line
            except ValueError:
                pass  # If the line does not start with a datetime, ignore the ValueError and process the line anyway
            new_logs.append(highlight_rules_func(line.rstrip('\n')))  # add the highlighted log to the list and strip any newline at the end
        last_position = f.tell()  # get the last position
    new_logs_as_string = "<br>".join(new_logs)  # joining with <br> directly
    return {"logs": new_logs_as_string, "last_position": last_position}  # also return the last position


def show_logs_func(minutes: int = 5):
    with open(log_file_path, "r") as f:
        lines = f.readlines()
    logs = []
    now = datetime.now(timezone('UTC'))
    for line in lines:
        if line.strip() == "":
            continue
        try:
            log_datetime_str = line.split(" - ")[0]
            log_datetime = datetime.strptime(log_datetime_str, "%Y-%m-%d %H:%M:%S,%f")
            log_datetime = log_datetime.replace(tzinfo=timezone('UTC'))
            if now - log_datetime <= timedelta(minutes=minutes):
                logs.append(highlight_rules_func(line.rstrip('\n')))
        except (ValueError, IndexError):
            logs.append(highlight_rules_func(line.rstrip('\n')))  # Line didn't meet datetime parsing criteria, continue with processing
    logs_as_string = "<br>".join(logs)
    logs_as_string_newlines_rendered = logs_as_string.replace("\n", "<br>")
    logs_as_string_newlines_rendered_font_specified = """
    <html>
    <head>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <script>
    var logContainer;
    var lastLogs = `{0}`;
    var shouldScroll = true;
    var userScroll = false;
    var lastPosition = 0;
    var minutes = {1};
    function fetchLogs() {{
        if (typeof minutes !== 'undefined' && typeof lastPosition !== 'undefined') {{
            fetch('/show_logs_incremental/' + minutes + '/' + lastPosition)
            .then(response => response.json())
            .then(data => {{
                if (logContainer) {{
                    var div = document.createElement('div');
                    div.innerHTML = data.logs;
                    if (div.innerHTML) {{
                        lastLogs += div.innerHTML;
                        lastPosition = data.last_position;
                    }}
                    logContainer.innerHTML = lastLogs;
                    if (shouldScroll) {{
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }}
                }}
            }});
        }}
    }}
    function checkScroll() {{
        if(logContainer.scrollTop + logContainer.clientHeight < logContainer.scrollHeight) {{
            userScroll = true;
            shouldScroll = false;
        }} else {{
            userScroll = false;
        }}
        if (!userScroll) {{
            setTimeout(function(){{ shouldScroll = true; }}, 10000);
        }}
    }}
    window.onload = function() {{
        let p = document.getElementsByTagName('p');
        for(let i = 0; i < p.length; i++) {{
            let color = window.getComputedStyle(p[i]).getPropertyValue('color');
            p[i].style.textShadow = `0 0 5px ${{color}}, 0 0 10px ${{color}}, 0 0 15px ${{color}}, 0 0 20px ${{color}}`;
        }}
        document.querySelector('#copy-button').addEventListener('click', function() {{
            var text = document.querySelector('#log-container').innerText;
            navigator.clipboard.writeText(text).then(function() {{
                console.log('Copying to clipboard was successful!');
            }}, function(err) {{
                console.error('Could not copy text: ', err);
            }});
        }});
        document.querySelector('#download-button').addEventListener('click', function() {{
            var text = document.querySelector('#log-container').innerText;
            var element = document.createElement('a');
            element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
            element.setAttribute('download', 'swiss_army_llama_monitor_log__' + new Date().toISOString() + '.txt');
            element.style.display = 'none';
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        }});
    }}
    document.addEventListener('DOMContentLoaded', (event) => {{
        logContainer = document.getElementById('log-container');
        logContainer.innerHTML = lastLogs;
        logContainer.addEventListener('scroll', checkScroll);
        fetchLogs();
        setInterval(fetchLogs, 20000);  // Fetch the logs every 20 seconds
    }});
    </script>
    </head>        
    <style>
    .log-container {{
        scroll-behavior: smooth;
        background-color: #2b2b2b; /* dark gray */
        color: #d3d3d3; /* light gray */
        background-image: linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px);
        background-size: 100% 10px, 10px 100%;
        background-position: 0 0, 0 0;
        animation: scan 1s linear infinite;
        @keyframes scan {{
            0% {{
                background-position: 0 0, 0 0;
            }}
            100% {{
                background-position: -10px -10px, -10px -10px;
            }}
        }}
        font-size: 14px;
        font-family: monospace;
        padding: 10px;
        height: 100vh;
        margin: 0;
        box-sizing: border-box;
        overflow: auto;
    }}
    .icon-button {{
        position: fixed;
        right: 10px;
        margin: 10px;
        background-color: #555;
        color: white;
        border: none;
        cursor: pointer;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
    }}
    #copy-button {{
        bottom: 80px;  // Adjust this value as needed
    }}
    #download-button {{
        bottom: 10px;
    }}
    </style>
    <body>
    <pre id="log-container" class="log-container"></pre>
    <button id="copy-button" class="icon-button"><i class="fas fa-copy"></i></button>
    <button id="download-button" class="icon-button"><i class="fas fa-download"></i></button>
    </body>
    </html>""".format(logs_as_string_newlines_rendered, minutes)
    return logs_as_string_newlines_rendered_font_specified