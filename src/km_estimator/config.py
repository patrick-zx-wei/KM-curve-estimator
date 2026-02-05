# Model names
GPT5_MINI_MODEL = "gpt-5-mini"
GEMINI_FLASH_MODEL = "gemini-3.0-flash"

# API defaults
API_TIMEOUT_SECONDS = 30
API_MAX_RETRIES = 3

# Pipeline retries
MAX_INPUT_GUARD_RETRIES = 3
MAX_MMPU_RETRIES = 3
MAX_VALIDATION_RETRIES = 3

# Thresholds
CONVERGENCE_THRESHOLD = 0.9
VALIDATION_MAE_THRESHOLD = 0.02

# Tiered extraction thresholds
TIERED_CONFIDENCE_THRESHOLD = 0.75
TIERED_SIMILARITY_THRESHOLD = 0.9

# Comparison tolerance for floating-point values
FLOAT_TOLERANCE = 0.01

# Cost per 1K tokens in USD
GPT5_MINI_COST_INPUT = 0.0001
GPT5_MINI_COST_OUTPUT = 0.0002
GEMINI_FLASH_COST_INPUT = 0.00025
GEMINI_FLASH_COST_OUTPUT = 0.0005

# Image resolution
TARGET_RESOLUTION = 2000
MIN_RESOLUTION = 200
MAX_RESOLUTION = 4000

# Quality thresholds
MIN_IMAGE_VARIANCE = 100.0
MIN_CONFIDENCE_OUTPUT = 0.3

# Extraction confidence defaults
DEFAULT_EXTRACTION_CONFIDENCE = 0.5
METADATA_MIN_CONFIDENCE_BOOST = 0.7

# Concurrency
DEFAULT_MAX_CONCURRENCY = 10

# Prompts
OCR_PROMPT = """Extract all text from this Kaplan-Meier survival curve image.

Return JSON with these fields:
- x_tick_labels: list of x-axis tick labels (e.g., ["0", "12", "24", "36"])
- y_tick_labels: list of y-axis tick labels (e.g., ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
- axis_labels: list of axis labels (e.g., ["Time (months)", "Survival probability"])
- legend_labels: list of legend/group names (e.g., ["Treatment", "Control"])
- risk_table_text: 2D array of risk table values if present, null otherwise
- title: plot title if present, null otherwise
- annotations: list of other text (p-values, hazard ratios, etc.)

Return only valid JSON, no markdown."""

OCR_PROMPT_WITH_CONFIDENCE = """Extract all text from this Kaplan-Meier survival curve image.

Return JSON with these fields:
- x_tick_labels: list of x-axis tick labels (e.g., ["0", "12", "24", "36"])
- y_tick_labels: list of y-axis tick labels (e.g., ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
- axis_labels: list of axis labels (e.g., ["Time (months)", "Survival probability"])
- legend_labels: list of legend/group names (e.g., ["Treatment", "Control"])
- risk_table_text: 2D array of risk table values if present, null otherwise
- title: plot title if present, null otherwise
- annotations: list of other text (p-values, hazard ratios, etc.)
- extraction_confidence: float 0.0-1.0 indicating your confidence in extraction accuracy

Return only valid JSON, no markdown."""

# Model-specific OCR prompts
# GPT: Conservative, focus on accuracy over quantity
OCR_PROMPT_GPT = """Extract all visible text from this Kaplan-Meier survival plot.

Focus on ACCURACY over quantity. Extract:
- x_tick_labels: X-axis tick values (time points)
- y_tick_labels: Y-axis tick values (survival probabilities)
- axis_labels: Axis title labels
- legend_labels: Legend entries (group/curve names)
- risk_table_text: Risk table as 2D array if visible, null otherwise
- title: Plot title if present, null otherwise
- annotations: Other text (p-values, hazard ratios, etc.)
- extraction_confidence: float 0.0-1.0 for your confidence

IMPORTANT: Check y_tick_labels carefully - the Y-axis may NOT start at 0.

Return only valid JSON, no markdown."""

# Gemini: Aggressive, push for completeness
OCR_PROMPT_GEMINI = """Extract ALL visible text from this Kaplan-Meier survival plot with HIGH DETAIL.

You MUST extract:
- x_tick_labels: ALL X-axis tick labels (every visible time value)
- y_tick_labels: ALL Y-axis tick labels (every visible probability value)
- axis_labels: Complete axis titles/labels
- legend_labels: ALL legend entries with exact text
- risk_table_text: COMPLETE risk table as 2D array (all rows, all columns, all numbers) or null
- title: Plot title if present, null otherwise
- annotations: ALL other text (p-values, HRs, confidence intervals, etc.)
- extraction_confidence: float 0.0-1.0 for your confidence

CRITICAL: The Y-axis may NOT start at 0.0 - check the actual lowest tick value.
Be thorough - extract EVERY piece of text visible in the image.

Return only valid JSON, no markdown."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this Kaplan-Meier survival curve image.

Previously extracted text from the image:
{ocr_json}

Using both the image and extracted text, return JSON with:
- x_axis: {{label, start, end, tick_interval, tick_values, scale}}
- y_axis: {{label, start, end, tick_interval, tick_values, scale}}
- curves: [{{name, color_description, line_style}}]
- risk_table: {{time_points, groups: [{{name, counts}}]}} or null
- title: string or null
- annotations: list of strings

CRITICAL - Y-AXIS ATTENTION:
- The Y-axis may NOT start at 0.0 - check the actual tick values carefully
- Common truncation: Y-axis starts at 0.2, 0.3, 0.4, or 0.5
- Set y_axis.start to the ACTUAL lowest tick mark value shown
- This is essential for accurate curve digitization

Return only valid JSON, no markdown."""

INPUT_GUARD_PROMPT = """Analyze this image and determine if it is a valid Kaplan-Meier survival curve.

Check for the following elements:
1. Axes: Are there clear X and Y axes?
2. Curves: Is there at least one step-function survival curve?
3. Ticks: Are there readable tick marks/labels on both axes?
4. Legend: Is there a legend identifying curve groups? (optional but helpful)
5. Risk table: Is there a number-at-risk table below the plot? (optional)

Return JSON with:
- valid: true if this is a usable KM curve image, false otherwise
- axes_present: true/false
- curves_present: true/false
- ticks_readable: true/false
- legend_present: true/false
- risk_table_present: true/false
- feedback: brief explanation of issues found (or "OK" if valid)

Return only valid JSON, no markdown."""
