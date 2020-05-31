import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tti-explorer",
    version="0.0.1",
    author="brynhayder, MJHutchinson, apaleyes, shehzaidi, bobby-he, ywteh",
    description="Code for experiments with TTI strategies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rs-delve/tti-explorer",
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    # The following may be untrue and needs to be checked!
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
