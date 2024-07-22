from dataclasses import dataclass
import vertexai
from vertexai.generative_models import GenerativeModel, Part, ChatSession, Image
import pandas as pd
import time
import unicodedata
from preprocess_vis30k import create_image_objects
import logging
