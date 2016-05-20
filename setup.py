from setuptools import setup

def readme():
	"""
	This function just return the content of README.md
	"""
	with open('README.md') as f:
		return f.read()

setup(name='ezyrb',
	  version='0.1',
	  description='POD',
	  long_description=readme(),
	  classifiers=[
	  	'Development Status :: 1 - Planning',
	  	'License :: OSI Approved :: MIT License',
	  	'Programming Language :: Python :: 2.7',
	  	'Intended Audience :: Science/Research',
	  	'Topic :: Scientific/Engineering :: Mathematics'
	  ],
	  keywords='dimension_reduction mathematics vtk pod',
	  url='https://github.com/mathLab/EZyRB',
	  author='Filippo Salmoiraghi, Marco Tezzele',
	  author_email='filippo.salmoiraghi@gmail.com, marcotez@gmail.com',
	  license='MIT',
	  packages=['ezyrb'],
	  install_requires=[
	  		'numpy',
	  		'scipy',
	  		'matplotlib',
	  		'enum34',
	  		'Sphinx>=1.4',
	  		'sphinx_rtd_theme'
	  ],
	  test_suite='nose.collector',
	  tests_require=['nose'],
	  include_package_data=True,
	  zip_safe=False)
