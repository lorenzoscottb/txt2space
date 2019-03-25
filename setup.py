from setuptools import setup
 
setup(
    name='txt2space',    # This is the name of your PyPI-package.
    version='0.1.2',                          # Update the version number for new releases
#    scripts=['helloworld']                  # The name of your scipt, and also the command you'll be using for calling it
    install_requires=['numpy', 'sklearn', 'nltk'],

)
