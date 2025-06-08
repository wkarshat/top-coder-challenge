from setuptools import setup, find_packages

setup(
    name="legacyr",
    version="0.1.0",
    description="Legacy Reimbursement System Analysis and Reverse Engineering",
    author="LegacyR Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 