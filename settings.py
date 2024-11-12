from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'CENAR.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'CENAR_detected.png'

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'

# Webcam
WEBCAM_PATH = 0
