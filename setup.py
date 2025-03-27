from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jetski-tracker",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="JetSki detection and tracking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jetski-tracking",
    packages=find_packages(),
    package_data={
        "jetski_tracker": ["py.typed"],
    },
    install_requires=[
        'ultralytics>=8.0.0',
        'opencv-python>=4.5.0',
        'numpy>=1.20.0',
        'PyYAML>=6.0',
        'requests>=2.25.0'
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-mock>=3.0.0',
            'mypy>=0.910',
            'flake8>=4.0.0',
            'black>=22.0.0',
            'sphinx>=4.0.0'
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="computer-vision object-detection jetski yolo",
)