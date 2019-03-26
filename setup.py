import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="txt2space-lorenzoscottb",
    version="0.1.2",
    author="Lorenzo Scott Bertolini",
    author_email="l.bertolini@sussex.ac.uk",
    description="convert txt file to vector space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lorenzoscottb/txt2space",
    packages=setuptools.find_packages(),
)
