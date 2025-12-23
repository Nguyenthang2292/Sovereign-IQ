# I Ching Module

Module for generating I Ching hexagrams and automating web form filling with result extraction.

## Overview

This module provides functionality to:
- Generate random I Ching hexagrams
- Create visual hexagram images
- Automate web form filling using Selenium
- Extract detailed information from result screenshots using Google Gemini API

## Structure

```
modules/iching/
├── cli/                    # Command line interface
│   ├── __init__.py
│   └── main.py            # Main entry point
├── core/                   # Core business logic
│   ├── __init__.py
│   ├── hexagram.py        # Hexagram generation logic
│   ├── image_generator.py # Image generation
│   ├── web_automation.py  # Selenium web automation
│   ├── result_extractor.py # Gemini API result extraction
│   └── data_models.py    # Data models (HaoInfo, IChingResult)
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── helpers.py         # Helper functions
├── images/                 # Output folder for images
└── __init__.py            # Module exports
```

## Usage

### Basic Usage

```python
from modules.iching.cli.main import main

# Run the full workflow
main(auto_close=False)
```

### Programmatic Usage

```python
from modules.iching.core.hexagram import prepare_hexagram
from modules.iching.core.web_automation import fill_web_form

# Generate hexagram
line_info = prepare_hexagram()

# Fill web form and extract results
fill_web_form(line_info, auto_close=True)
```

### Using Individual Components

```python
from modules.iching.core.hexagram import generate_ns_string, group_string, analyze_line
from modules.iching.core.image_generator import create_hexagram_image
from modules.iching.core.result_extractor import IChingResultExtractor

# Generate random string
random_string = generate_ns_string(length=18)

# Group into hexagram lines
grouped = group_string(random_string)

# Analyze each line
for group in grouped:
    is_solid, is_red = analyze_line(group)

# Create hexagram image
image_path = create_hexagram_image(grouped)

# Extract results from image
extractor = IChingResultExtractor()
result = extractor.extract_from_image("path/to/image.png")
```

## Data Models

### HaoInfo

Represents information about a single line (hào) in a hexagram:

```python
@dataclass
class HaoInfo:
    hao: int                    # Line number (1-6, bottom to top)
    luc_than: str               # Six Relations (Thế, Ứng, etc.)
    can_chi: str                # Heavenly Stems and Earthly Branches
    luc_thu: Optional[str]      # Six Animals (only in right hexagram)
    is_dong: bool = False       # Dynamic line (red color)
    phuc_than: Optional[str]    # Hidden God (only in left hexagram)
```

### IChingResult

Represents the complete extracted result:

```python
@dataclass
class IChingResult:
    nhat_than: str              # Day God
    nguyet_lenh: str            # Month Order
    que_trai: List[HaoInfo]     # Left hexagram (6 lines)
    que_phai: List[HaoInfo]     # Right hexagram (6 lines)
    the_vi_tri: int             # Position of Thế (1-6)
    ung_vi_tri: int             # Position of Ứng (1-6)
    tk_vi_tri: Optional[List[int]]  # Positions of TK
```

## Configuration

Configuration is stored in `config/iching.py`:

- `HEXAGRAM_STRING_LENGTH`: Length of random string (default: 18)
- `GROUP_SIZE`: Size of each group (default: 3)
- `NUM_LINES`: Number of lines in hexagram (default: 6)
- `ICHING_URL`: URL of the I Ching web form
- Image generation parameters (width, height, positions, etc.)

## Dependencies

- `selenium`: Web automation
- `webdriver-manager`: ChromeDriver management (optional)
- `PIL` (Pillow): Image generation
- `modules.gemini_chart_analyzer`: For result extraction

## Testing

Run tests with pytest:

```bash
pytest tests/iching/
```

Test coverage includes:
- Core hexagram generation logic
- Image generation
- Data model serialization
- Result extraction (mocked)
- Web automation (mocked)
- Utility functions

## Notes

- The `images/` folder is automatically cleaned at the start of each run
- Screenshots and extracted JSON results are saved in the `images/` folder
- Dynamic lines (hào động) in the right hexagram are automatically synced with the left hexagram

