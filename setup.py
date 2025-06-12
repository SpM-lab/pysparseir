#!/usr/bin/env python3
import os
import re
import subprocess
import shutil
import urllib.request
import tarfile
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from setuptools.command.build_ext import build_ext

# Version settings
LIBSPARSEIR_VERSION = "v0.2.0"
LIBSPARSEIR_URL = "https://github.com/SpM-lab/libsparseir.git"
EIGEN3_VERSION = "3.4.0"
XPREC_VERSION = "0.7.0"

# C-API build settings
CAPI_SOURCES = [
    'libsparseir/src/cinterface.cpp',
    'libsparseir/src/kernel.cpp',
    'libsparseir/src/linalg.cpp',
    'libsparseir/src/poly.cpp',
    'libsparseir/src/root.cpp',
    'libsparseir/src/specfuncs.cpp',
    'libsparseir/src/svd.cpp',
    'libsparseir/src/sve.cpp',
    'libsparseir/src/utils.cpp',
]

CAPI_INCLUDE_DIRS = [
    'libsparseir/include',
    'libsparseir/deps/eigen3',
    'libsparseir/deps/xprec/include',
]

CAPI_EXTRA_COMPILE_ARGS = [
    '-std=c++17',
    '-O3',
    '-DNDEBUG',
]

if os.name == 'nt':  # Windows
    CAPI_EXTRA_COMPILE_ARGS.extend(['/EHsc'])
else:  # Linux/Unix/macOS
    CAPI_EXTRA_COMPILE_ARGS.extend(['-fPIC'])


def download_and_extract_tarball(url, target_dir, extract_dir):
    """Download and extract a tarball."""
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Download tarball
    tarball_path = os.path.join(target_dir, os.path.basename(url))
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, tarball_path)
    
    # Extract tarball
    with tarfile.open(tarball_path, "r:gz") as tar:
        # Get the root directory name before extraction
        root_dir = tar.getmembers()[0].name.split('/')[0]
        # Extract with filter to handle warning
        tar.extractall(path=target_dir, filter='data')
    
    # Move contents to target directory
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    src_dir = os.path.join(target_dir, root_dir)
    shutil.move(src_dir, extract_dir)
    
    # Clean up tarball
    os.remove(tarball_path)


def download_and_extract_tag(tag, target_dir):
    """Download and extract a specific tag from GitHub using git."""
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Clone specific tag
    libsparseir_dir = os.path.join(target_dir, "libsparseir")
    if os.path.exists(libsparseir_dir):
        shutil.rmtree(libsparseir_dir)
    
    print(f"Cloning {LIBSPARSEIR_URL} (tag: {tag})...")
    subprocess.check_call([
        'git', 'clone', '--depth', '1', '--branch', tag,
        LIBSPARSEIR_URL, libsparseir_dir
    ])


def ensure_dependencies():
    """Ensure all dependencies are downloaded and extracted."""
    pkg_dir = os.path.dirname(__file__)
    libsparseir_dir = os.path.join(pkg_dir, 'libsparseir')
    
    # Create deps directory inside libsparseir
    deps_dir = os.path.join(libsparseir_dir, 'deps')
    os.makedirs(deps_dir, exist_ok=True)
    
    # Download and extract Eigen3
    eigen3_dir = os.path.join(deps_dir, 'eigen3')
    if not os.path.exists(eigen3_dir):
        eigen3_url = (
            f"https://gitlab.com/libeigen/eigen/-/archive/"
            f"{EIGEN3_VERSION}/eigen-{EIGEN3_VERSION}.tar.gz"
        )
        download_and_extract_tarball(eigen3_url, deps_dir, eigen3_dir)
    
    # Download and extract xprec
    xprec_dir = os.path.join(deps_dir, 'xprec')
    if not os.path.exists(xprec_dir):
        xprec_url = (
            f"https://github.com/tuwien-cms/libxprec/archive/refs/tags/"
            f"v{XPREC_VERSION}.tar.gz"
        )
        download_and_extract_tarball(xprec_url, deps_dir, xprec_dir)
    
    return deps_dir


def ensure_libsparseir():
    """Ensure libsparseir is downloaded and extracted."""
    pkg_dir = os.path.dirname(__file__)
    libsparseir_dir = os.path.join(pkg_dir, 'libsparseir')
    if not os.path.exists(libsparseir_dir):
        download_and_extract_tag(LIBSPARSEIR_VERSION, pkg_dir)
        # Download dependencies after cloning libsparseir
        ensure_dependencies()
        
        # Copy Makefile.bundle to libsparseir directory
        #makefile_src = os.path.join(pkg_dir, 'Makefile.bundle')
        #makefile_dst = os.path.join(libsparseir_dir, 'Makefile.bundle')
        #if os.path.exists(makefile_src):
            #shutil.copy2(makefile_src, makefile_dst)
        #else:
            #raise RuntimeError("Makefile.bundle not found in package root")
            
    return libsparseir_dir


class BuildCommand(build_py):
    def run(self):
        # Download and extract libsparseir and dependencies
        libsparseir_dir = ensure_libsparseir()
        
        # Copy Makefile.bundle to libsparseir directory
        pkg_dir = os.path.dirname(__file__)
        makefile_src = os.path.join(pkg_dir, 'Makefile.bundle')
        makefile_dst = os.path.join(libsparseir_dir, 'Makefile.bundle')
        if os.path.exists(makefile_src):
            shutil.copy2(makefile_src, makefile_dst)
        else:
            raise RuntimeError("Makefile.bundle not found in package root")
        
        # Build the C library using Makefile.bundle
        print("Building C library...")
        make_cmd = ['make', '-f', 'Makefile.bundle']
        subprocess.check_call(make_cmd, cwd=libsparseir_dir)
        
        # Run the original build_py
        build_py.run(self)


class SDistCommand(sdist):
    def run(self):
        # Download and extract libsparseir and dependencies
        libsparseir_dir = ensure_libsparseir()
        
        # Copy Makefile.bundle to libsparseir directory
        pkg_dir = os.path.dirname(__file__)
        makefile_src = os.path.join(pkg_dir, 'Makefile.bundle')
        makefile_dst = os.path.join(libsparseir_dir, 'Makefile.bundle')
        if os.path.exists(makefile_src):
            shutil.copy2(makefile_src, makefile_dst)
        else:
            raise RuntimeError("Makefile.bundle not found in package root")
        
        # Ensure dependencies are downloaded
        deps_dir = os.path.join(libsparseir_dir, 'deps')
        if not os.path.exists(deps_dir):
            ensure_dependencies()
        
        # Run the original sdist
        sdist.run(self)


class BuildExtCommand(build_ext):
    def run(self):
        # Download and extract libsparseir and dependencies
        libsparseir_dir = ensure_libsparseir()
        
        # Copy Makefile.bundle to libsparseir directory
        pkg_dir = os.path.dirname(__file__)
        makefile_src = os.path.join(pkg_dir, 'Makefile.bundle')
        makefile_dst = os.path.join(libsparseir_dir, 'Makefile.bundle')
        if os.path.exists(makefile_src):
            shutil.copy2(makefile_src, makefile_dst)
        else:
            raise RuntimeError("Makefile.bundle not found in package root")
        
        # Build the C library using Makefile.bundle
        print("Building C library...")
        make_cmd = ['make', '-f', 'Makefile.bundle']
        subprocess.check_call(make_cmd, cwd=libsparseir_dir)
        
        # Run the original build_ext
        build_ext.run(self)


def get_version():
    # Ensure libsparseir is downloaded before getting version
    ensure_libsparseir()
    
    version_h = os.path.join(
        os.path.dirname(__file__),
        'libsparseir', 'include', 'sparseir', 'version.h'
    )
    with open(version_h, 'r') as f:
        content = f.read()
    
    major_match = re.search(r'#define SPARSEIR_VERSION_MAJOR (\d+)', content)
    minor_match = re.search(r'#define SPARSEIR_VERSION_MINOR (\d+)', content)
    patch_match = re.search(r'#define SPARSEIR_VERSION_PATCH (\d+)', content)
    
    if not all([major_match, minor_match, patch_match]):
        raise RuntimeError("Could not find version information in version.h")
    
    major = major_match.group(1)
    minor = minor_match.group(1)
    patch = patch_match.group(1)
    
    return f"{major}.{minor}.{patch}"


# Define C-API extension
sparseir_capi = Extension(
    'sparseir._capi',
    sources=CAPI_SOURCES,
    include_dirs=CAPI_INCLUDE_DIRS,
    extra_compile_args=CAPI_EXTRA_COMPILE_ARGS,
    language='c++',
)


setup(
    name="sparseir",
    version=get_version(),
    description="Python bindings for libsparseir",
    author="Hiroshi Shinaoka",
    author_email="h.shinaoka@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    ext_modules=[sparseir_capi],
    cmdclass=dict(
        build_py=BuildCommand,
        sdist=SDistCommand,
        build_ext=BuildExtCommand,
    ),
    package_data={
        'sparseir': [
            'libsparseir/deps/**/*',
            'libsparseir/include/**/*',
            'libsparseir/src/**/*',
            'libsparseir/Makefile.bundle',
            'libsparseir/build_capi.sh',
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics"
    ],
) 