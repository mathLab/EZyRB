# EZyRB[![Build Status](https://travis-ci.org/mathLab/EZyRB.svg)](https://travis-ci.org/mathLab/EZyRB) [![Coverage Status](https://coveralls.io/repos/github/mathLab/EZyRB/badge.svg?branch=master)](https://coveralls.io/github/mathLab/EZyRB?branch=master) [![Code Issues](https://www.quantifiedcode.com/api/v1/project/117cf59ced024d8cb09a9131a9c3bec1/badge.svg)](https://www.quantifiedcode.com/app/project/117cf59ced024d8cb09a9131a9c3bec1)
Easy Reduced Basis method.

![Easy Reduced Basis method](readme/logo_EZyRB_small.png)

## Description
**EZyRB** is a python library for the Model Order Reduction based on **baricentric triangulation** for the selection of the parameter points and on **Proper Orthogonal Decomposition** for the selection of the modes. It is ideally suited for actual industrial problems, since its structure can interact with several simulation software simply providing the output file of the simulations. Up to now, it handles files in the vtk and mat formats. It has been used for the model order reduction of problems solved with matlab and openFOAM.

See the **Examples** section below to have an idea of the potential of this package.

## Graphical User Interface
**EZyRB** is now provided with a very basic Graphical User Interface (GUI) that, in Ubuntu environment, looks like the one depicted below. This feature can be easily used even by the pythonists beginners with not much effort. Just see the [tutorial 3](https://github.com/mathLab/EZyRB/blob/master/tutorials/tutorial-3.ipynb). For the laziest, also a [video tutorial on YouTube](https://youtu.be/PPlRHu53VRA) is available.

<p align="center">
<img src="readme/EZyRB_gui.png" alt>
</p>
<p align="center">
<em>EZyRB GUI: how it appears when it pops up.</em>
</p>

Up to now, EZyRB GUI works on linux and Mac OS X computers.

## Dependencies and installation
**EZyRB** requires `numpy`, `scipy` and `matplotlib`. They can be easily installed via `pip`. Moreover **EZyRB** depends on `vtk`. These requirements cannot be satisfied through `pip`.
Please see the table below for instructions on how to satisfy the requirements.

| Package | Version  | Comment                                                                    |
|---------|----------|----------------------------------------------------------------------------|
| vtk     | >= 5.0   | Simplest solution is `conda install vtk`                                   |

The official distribution is on GitHub, and you can clone the repository using

```bash
> git clone https://github.com/mathLab/EZyRB
```

To install the package just type:

```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```


## Documentation
**EZyRB** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. To build the html versions of the docs simply:

```bash
> cd docs
> make html
```

The generated html can be found in `docs/build/html`. Open up the `index.html` you find there to browse.


## Testing
We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mathLab/EZyRB).

To run tests locally:

```bash
> python test.py
```


## Authors and contributors
**EZyRB** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by
* [Filippo Salmoiraghi](mailto:filippo.salmoiraghi@gmail.com)
* [Marco Tezzele](mailto:marcotez@gmail.com)

under the supervision of [Prof. Gianluigi Rozza](mailto:gianluigi.rozza@sissa.it).

Contact us by email for further information or questions about **EZyRB**, or suggest pull requests. **EZyRB** is at an early development stage, so contributions improving either the code or the documentation are welcome!

## Examples

You can find useful tutorials on how to use the package in the `tutorials` folder.
Here we show an applications taken from the **automotive** engineering fields

<p align="center">
<img src="readme/pod_modes.png" alt>
</p>
<p align="center">
<em>The first POD modes of the pressure field on the DrivAer model.</em>
</p>

<p align="center">
<img src="readme/errors.png" alt>
</p>
<p align="center">
<em>DrivAer model online evaluation: pressure (left) and wall shear stress (right) fields and errors.</em>
</p>

## How to contribute
We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use a tab of 4
     spaces to indent the code. You should also run [pylint][] over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.

  4. Any significant changes should almost always be accompanied by tests.  The
     project already has good test coverage, so look at some of the existing
     tests if you're unsure how to go about it. We're using [coveralls][] that
     is an invaluable tools for seeing which parts of your code aren't being
     exercised by your tests.

  5. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  6. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pylint]: https://www.pylint.org/
[coveralls]: https://coveralls.io
[well-formed commit messages]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[pull request]: https://help.github.com/articles/creating-a-pull-request


## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
