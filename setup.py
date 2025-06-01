from setuptools import setup, find_packages

setup(
    name="dataquality",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scikit-learn","azure-ai-anomalydetector"],
    author="WetMix",
    description="Library for analyzing data quality in time series.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    package_data={
        'dataquality': ['solar_sample.csv'],
    },
)
