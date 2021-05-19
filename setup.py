from setuptools import setup, Command
import os
import sys
from shutil import rmtree

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
KEYWORDS='pod interpolation reduced-basis model-order-reduction'

REQUIRED = [
    'future', 'numpy', 'scipy',	'matplotlib', 'GPy', 'sklearn', 'torch'
]

EXTRAS = {
    'docs': ['Sphinx==1.4', 'sphinx_rtd_theme'],
}

LDESCRIPTION = (
    "EZyRB is a python library for the Model Order Reduction based on "
    "baricentric triangulation for the selection of the parameter points and on "
    "Proper Orthogonal Decomposition for the selection of the modes. It is "
    "ideally suited for actual industrial problems, since its structure can "
    "interact with several simulation software simply providing the output file "
    "of the simulations."
)

here = os.path.abspath(os.path.dirname(__file__))
class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()

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
	packages=[NAME],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    test_suite='nose.collector',
	tests_require=['nose'],
	include_package_data=True,
	zip_safe=False,

    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },)
