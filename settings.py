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
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, WEBCAM, VIDEO]  

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'CENAR.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'CENAR_detected.png'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'ComLab1.mp4',
    'video_2': VIDEO_DIR / 'video_2.mp4',
    'video_3': VIDEO_DIR / 'video_3.mp4',
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'first.pt'

# Webcam
WEBCAM_PATH = 0
