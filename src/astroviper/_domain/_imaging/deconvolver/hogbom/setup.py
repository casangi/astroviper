from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "hogbom_clean",
        [
            "python/bindings.cpp",
            "src/hogbom_clean.cpp",
        ],
        include_dirs=[
            "include",
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
        define_macros=[('VERSION_INFO', '"0.1.0"')],
    ),
]

# Compiler-specific options
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-O2', '-march=native'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if os.environ.get('DEBUG', '0') == '1':
        c_opts['unix'] = ['-O0', '-g']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            if hasattr(self.compiler, 'compiler_so'):
                if 'clang' in self.compiler.compiler_so[0]:
                    opts.append('-stdlib=libc++')
                    link_opts.append('-stdlib=libc++')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
