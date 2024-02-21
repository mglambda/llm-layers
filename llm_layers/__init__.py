from llm_layers.layers import get_hf_repo_for_file, load_layers_file, download_for_layer_file, get_total_vram_mb
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def getData(path):
    return os.path.join(_ROOT, 'data', path)




