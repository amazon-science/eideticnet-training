from setuptools import setup, find_packages

setup(
    name="eideticnet_training",
    version="0.1.0",
    description="Package for training EideticNets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nicholas Dronen",
    author_email="ndronen@amazon.com",
    url="https://github.com/amazon-science/eideticnet",
    license="Apache Public License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="PyTorch, neural networks, continuous learning",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        "wandb",
        "pandas",
        "tqdm",
        "ipython",
        "datasets",
    ],
    extras_require={
        "dev": ["check-manifest", "black", "flake8", "pre-commit"],
        "test": ["pytest", "coverage"],
    },
)
