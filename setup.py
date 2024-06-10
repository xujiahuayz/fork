from setuptools import setup
from Cython.Build import cythonize
import numpy
from fork_env.settings import PROJECT_ROOT

setup(
    ext_modules=cythonize(str(PROJECT_ROOT/"fork_env/integrationCy.pyx")),
    include_dirs=[numpy.get_include()],
)
