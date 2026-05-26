"""Setup configuration for polyglot-er."""

from setuptools import setup, find_packages

setup(
    name="polyglot-er",
    version="0.1.0",
    author="Daniel Schmidt",
    description="Cross-Lingual Entity Resolution Pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/polyglot-er",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "jellyfish>=0.11.2",
        "requests>=2.28",
    ],
    extras_require={
        "embeddings": [
            "sentence-transformers>=2.2.0",
            "numpy>=1.24",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "responses>=0.23",
        ],
    },
    entry_points={
        "console_scripts": [
            "polyglot-er=cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
