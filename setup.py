from setuptools import setup, find_packages

ver = {}
try:
    with open('Utilities/_version.py') as fd:
        exec(fd.read(), ver)
    version = ver.get('__version__', 'dev')
except IOError:
    version = 'dev'

with open('README.rst') as fp:
    long_description = fp.read()

CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

setup(
    name='levelsetpy',
    version=version,
    author='Lekan Molu',
    author_email='lekanmolu@microsoft.com',
    url='http://scriptedonachip.com',
    description='Level Set Methods in Python Library',
    long_description=long_description,
    packages=find_packages(exclude=['benchmarks']),
    classifiers=[f for f in CLASSIFIERS.split('\n') if f],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib'],
    extras_require={
       'test': ['pytest', 'pytest-timeout'],
       'slycot': [ 'slycot>=0.4.0' ]
    }
)
