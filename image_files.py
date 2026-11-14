"""
Machine-independent listing of image files.

A raw directory listing is not the same everywhere: NTFS returns names sorted,
ext4 returns them in hash order, and a case-insensitive file system matches
"*.jpg" and "*.JPG" to the same files. Training splits and the pairing of test
images with seeded rotation angles both depend on the order of the list, so
every listing goes through this module. It has no heavy dependencies, so the
evaluation scripts of other environments can import it too.
"""
import warnings
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
TEST_ORDER_DIR = Path(__file__).resolve().parent / "eval" / "test_order"


def list_image_files(image_dir):
    """All image files of a directory as Path objects, sorted by name."""
    return sorted((p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: p.name)


def list_test_images(test_dir):
    """
    The test images, as path strings, in the order that pairs them with the
    seeded test angles.

    Alphabetical, unless eval/test_order/<directory name>.txt pins another order.
    The paper's test sets (test_coco, test_drcd) were evaluated in the directory
    order of the machine the experiments ran on; the pinned lists reproduce that
    pairing, and with it the published numbers, on any machine. A pinned list is
    used only if it names exactly the files that are present.
    """
    paths = list_image_files(test_dir)
    order_file = TEST_ORDER_DIR / f"{Path(test_dir).name}.txt"
    if order_file.exists():
        names = order_file.read_text().split()
        by_name = {p.name: p for p in paths}
        if len(names) == len(by_name) and set(names) == set(by_name):
            return [str(by_name[n]) for n in names]
        warnings.warn(f"{order_file} does not match the files in {test_dir}; using alphabetical order")
    return [str(p) for p in paths]
