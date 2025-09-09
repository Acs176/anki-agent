import os
import sys

from dotenv import load_dotenv


def _add_src_to_path():
    # Ensure `src` is importable when running tests without installing the package
    here = os.path.dirname(__file__)
    src_path = os.path.abspath(os.path.join(here, "..", "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


load_dotenv()
_add_src_to_path()
