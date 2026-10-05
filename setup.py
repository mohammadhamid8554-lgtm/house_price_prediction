from pathlib import Path
from typing import List

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
HYPHEN_E_DOT = "-e ."


def get_requirement(file_path: Path) -> List[str]:
    with file_path.open(encoding="utf-8") as file_obj:
        requirements = [line.strip() for line in file_obj if line.strip()]

    return [requirement for requirement in requirements if requirement != HYPHEN_E_DOT]


setup(
    name="house_price_prediction",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirement(ROOT / "requirements.txt"),
)
