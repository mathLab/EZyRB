from setuptools import setup, find_packages

meta = {}
with open("ezyrb/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta['__title__']
DESCRIPTION = 'Easy Reduced Basis'
URL = 'https://github.com/mathLab/EZyRB'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = 'pod interpolation reduced-basis model-order-reduction'

REQUIRED = [
    'future', 'numpy', 'scipy',	'matplotlib', 'scikit-learn>=1.0', 'torch'
]

EXTRAS = {
    'docs': ['sphinx', 'sphinx_rtd_theme'],
    'test': ['pytest', 'pytest-cov'],
}

LDESCRIPTION = (
    "EZyRB is a Python package that performs a data-driven model order "
    "reduction for parametrized problems exploiting the recent approaches. "
    "Such techniques are able to provide a parametric model capable to "
    "provide the real-time approximation of the solution of a generic "
    "(potentially complex and non linear) problem. The reduced model is "
    "totally built upon the numerical data obtained by the original (to "
    "reduce) model, without requiring any knowledge of the equations that "
    "describe this model, resulting in a well suited framework for industrial "
    "contexts due to its natural capability to been integrated with "
    "commercial software."
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    license='MIT',
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
)
