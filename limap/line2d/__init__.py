import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .register_detector import get_detector, get_extractor
from .register_matcher import get_matcher

