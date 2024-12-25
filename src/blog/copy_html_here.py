"""Copy HTML files from the project's figures directory to the current directory."""

import shutil
from pathlib import Path

from loguru import logger

from src.utils.general_setup import setup


setup("copy_html_here")

PROJ_ROOT = Path(__file__).parent.parent.parent


def copy_html_files() -> None:
    """Copy all HTML files from PROJ_ROOT/data/figs/html to the current directory.

    :param Path proj_root: Path to project root. If None, attempts to find it by traversing up
    :return: None
    :raises FileNotFoundError: If project root or source directory cannot be found
    """
    # Define source and destination directories
    src_dir = PROJ_ROOT / "data" / "figs" / "html"
    target_dir = Path(__file__).parent / "figs"

    # Verify source directory exists
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy all HTML files
    html_files = list(src_dir.glob("*.html"))
    if not html_files:
        logger.warning(f"No HTML files found in {src_dir}")
        return

    for html_file in html_files:
        dest_file = target_dir / html_file.name
        shutil.copy2(html_file, dest_file)
        logger.info(f"Copied {html_file.name} to {target_dir}")


if __name__ == "__main__":
    try:
        copy_html_files()
    except Exception as e:
        logger.error(f"Error copying HTML files: {e}")
        raise
