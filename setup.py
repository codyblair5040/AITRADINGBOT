from setuptools import setup, find_packages

setup(
    name="enterprise-trading-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.0.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
)
