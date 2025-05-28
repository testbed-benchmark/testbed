import re
from os import path

from setuptools import find_packages, setup

name = 'testbed'
here = path.abspath(path.dirname(__file__))

# get package version
with open(path.join(here, name, "__init__.py"), encoding="utf-8") as f:
    result = re.search(r'__version__ = ["\']([^"\']+)', f.read())
    if not result:
        raise ValueError("Can't find the version in causalnex/__init__.py")
    version = result.group(1)

setup(
    name=name,
    version=version,
    packages=find_packages(),
)
