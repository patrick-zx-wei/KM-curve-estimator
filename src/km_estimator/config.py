# Model names
GEMINI_PRO_MODEL = "gemini-3-pro-preview"
GEMINI_FLASH_MODEL = "gemini-3-flash-preview"

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

# Image resolution
TARGET_RESOLUTION = 2000
MIN_RESOLUTION = 200
MAX_RESOLUTION = 4000

# Quality thresholds
MIN_IMAGE_VARIANCE = 100.0
MIN_CONFIDENCE_OUTPUT = 0.3

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

Return only valid JSON, no markdown."""
