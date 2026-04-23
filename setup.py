from setuptools import setup, find_packages

setup(
    name="house-price-prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "joblib",
        "pandas",
        "scikit-learn",
        "pytz",
        "redis",
        "fastapi",
        "uvicorn",
    ],
    include_package_data=True,
)