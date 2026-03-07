import glob
import re

files_to_check = glob.glob('modules/auto_trade/gui/**/*.py', recursive=True)

color_map = {
    r'fg_color=[\"\'](?:#00ff88|#10b981|#00aa66|#22cc88)[\"\']': 'fg_color=Colors.BTN_SUCCESS',
    r'hover_color=[\"\'](?:#00cc66|#008855|#059669|#11aa66)[\"\']': 'hover_color=Colors.BTN_SUCCESS_HOVER',
    r'fg_color=[\"\'](?:#ff4444|#ff6644|#e53e3e)[\"\']': 'fg_color=Colors.BTN_DANGER',
    r'hover_color=[\"\'](?:#cc0000|#cc4422|#c53030)[\"\']': 'hover_color=Colors.BTN_DANGER_HOVER',
    r'fg_color=[\"\'](?:#7f1d1d)[\"\']': 'fg_color=Colors.BTN_DANGER_ALT',
    r'hover_color=[\"\'](?:#991b1b)[\"\']': 'hover_color=Colors.BTN_DANGER_ALT_HOVER',
    r'fg_color=[\"\'](?:#4488ff|#00aaff|#1f538d|#2b6cb0|#1f6aa5)[\"\']': 'fg_color=Colors.BTN_PRIMARY',
    r'hover_color=[\"\'](?:#0066ff|#0088cc|#2266cc|#3182ce|#2a6bb5|#144870)[\"\']': 'hover_color=Colors.BTN_PRIMARY_HOVER',
    r'fg_color=[\"\'](?:#555555|#666666|#44aa88|#aa44ff|#444444|#4a5568|#556677|#556677)[\"\']': 'fg_color=Colors.BTN_NEUTRAL',
    r'hover_color=[\"\'](?:#333333|#777777|#444444|#338866|#8822cc|#2d3748|#445566)[\"\']': 'hover_color=Colors.BTN_NEUTRAL_HOVER',
    r'fg_color=[\"\'](?:#ffaa00|#ff8844|#dd6b20)[\"\']': 'fg_color=Colors.BTN_WARNING',
    r'hover_color=[\"\'](?:#cc8800|#cc6622|#c05621)[\"\']': 'hover_color=Colors.BTN_WARNING_HOVER',
}

for f in files_to_check:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    original_content = content
    for pattern, replacement in color_map.items():
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
    if content != original_content:
        # Check if Colors is imported
        if 'import Colors' not in content and 'Colors.' in content:
            if 'import customtkinter as ctk' in content:
                content = content.replace('import customtkinter as ctk', 'import customtkinter as ctk\nfrom modules.auto_trade.gui.utils.colors import Colors')
            elif 'import customtkinter' in content:
                content = content.replace('import customtkinter', 'import customtkinter\nfrom modules.auto_trade.gui.utils.colors import Colors')
            else:
                content = 'from modules.auto_trade.gui.utils.colors import Colors\n' + content
        
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated {f}")
