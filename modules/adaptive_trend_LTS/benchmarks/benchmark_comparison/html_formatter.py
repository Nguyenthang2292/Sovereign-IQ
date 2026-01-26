"""HTML formatter for converting ANSI color codes to HTML."""

import re


def ansi_to_html(text: str) -> str:
    """Convert ANSI color codes to HTML with colored spans.

    Maps colorama ANSI codes to HTML:
    - Fore.BLUE -> blue
    - Fore.GREEN -> green
    - Fore.RED -> red
    - Fore.YELLOW -> yellow
    - Style.BRIGHT -> bold
    - Style.RESET_ALL -> close span

    Args:
        text: Text containing ANSI escape sequences

    Returns:
        HTML string with colored spans and dark theme styling
    """
    # ANSI escape sequence pattern: \x1b[<code>m
    ansi_pattern = re.compile(r'\x1b\[(\d+)(?:;(\d+))?m')

    # Color code mappings (30-37 are foreground colors, 1 is bold)
    color_map = {
        '30': '#000000',  # Black
        '31': '#f48771',  # Red
        '32': '#4ec9b0',  # Green
        '33': '#dcdcaa',  # Yellow
        '34': '#569cd6',  # Blue
        '35': '#c586c0',  # Magenta
        '36': '#4ec9b0',  # Cyan
        '37': '#d4d4d4',  # White
    }

    result = []
    i = 0
    open_tags = []  # Track open HTML tags

    while i < len(text):
        # Check for ANSI escape sequence
        match = ansi_pattern.match(text, i)
        if match:
            code1 = match.group(1)
            code2 = match.group(2)

            # Handle reset (0) - close all open tags
            if code1 == '0':
                # Close all open tags in reverse order
                for tag in reversed(open_tags):
                    result.append(f'</{tag}>')
                open_tags = []

            # Handle bold (1)
            elif code1 == '1':
                if 'strong' not in open_tags:
                    result.append('<strong>')
                    open_tags.append('strong')

            # Handle foreground colors (30-37)
            elif code1 in color_map:
                # Close previous color span if exists
                if open_tags and open_tags[-1] == 'span':
                    result.append('</span>')
                    open_tags.pop()

                color = color_map[code1]
                result.append(f'<span style="color: {color};">')
                open_tags.append('span')

            # Handle combined codes (e.g., "1;31" for bold red)
            if code2:
                if code2 == '1' and 'strong' not in open_tags:
                    result.append('<strong>')
                    open_tags.append('strong')
                elif code2 in color_map:
                    if open_tags and open_tags[-1] == 'span':
                        result.append('</span>')
                        open_tags.pop()
                    color = color_map[code2]
                    result.append(f'<span style="color: {color};">')
                    open_tags.append('span')

            i = match.end()
        else:
            # Regular character - escape HTML and add
            char = text[i]
            if char == '&':
                result.append('&amp;')
            elif char == '<':
                result.append('&lt;')
            elif char == '>':
                result.append('&gt;')
            else:
                result.append(char)
            i += 1

    # Close any remaining open tags
    for tag in reversed(open_tags):
        result.append(f'</{tag}>')

    html_text = ''.join(result)

    # Wrap in HTML document with dark theme
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Benchmark Log</title>
    <style>
        body {{
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            margin: 0;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
            background-color: #252526;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #3e3e42;
        }}
        strong {{
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <pre>{html_text}</pre>
</body>
</html>"""

    return html_content
