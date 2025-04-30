Guide for Contributors
============================

We welcome everyone’s contributions! The development of TensorCircuit is open-sourced and centered on `GitHub <https://github.com/tencent-quantum-lab/tensorcircuit>`_.

There are various ways to contribute:

* Answering questions on the discussions page or issue page.

* Raising issues such as bug reports or feature requests on the issue page.

* Improving the documentation (docstrings/tutorials) by pull requests.

* Contributing to the codebase by pull requests.



Pull Request Guidelines
-------------------------------

We welcome pull requests from everyone. For large PRs involving feature enhancement or API changes, we ask that you first open a GitHub issue to discuss your proposal.

The following git workflow is recommended for contribution by PR:

* Configure your git username and email so that they match your GitHub account if you haven't.

.. code-block:: bash

    git config user.name <GitHub name>
    git config user.email <GitHub email>

* Fork the TensorCircuit repository by clicking the Fork button on GitHub. This will create an independent version of the codebase in your own GitHub account.

* Clone your forked repository and set up an ``upstream`` reference to the official TensorCircuit repository.

.. code-block:: bash

    git clone <your-forked-repo-git-link>
    cd tensorcircuit
    git remote add upstream <official-repo-git-link>

* Configure the python environment locally for development. The following commands are recommended:

.. code-block:: bash

    pip install -r requirements/requirements.txt
    pip install -r requirements/requirements-dev.txt

Extra packages may be required for specific development tasks.

* Pip installing your fork from the source. This allows you to modify the code locally and immediately test it out.

.. code-block:: bash

    python setup.py develop

* Create a feature branch where you can make modifications and developments. DON'T open PR from your master/main branch.

.. code-block:: bash

    git checkout -b <name-of-change>

* Make sure your changes can pass all checks by running: ``./check_all.sh``. (See the :ref:`Checks` section below for details)

* Once you are satisfied with your changes, create a commit as follows:

.. code-block:: bash

    git add file1.py file2.py ...
    git commit -m "Your commit message (should be brief and informative)"
    
* You should sync your code with the official repo:

.. code-block:: bash

    git fetch upstream
    git rebase upstream/master      # resolve conflicts if any

* Note that PRs typically comprise a single git commit, you should squash all your commits in the feature branch. Using ``git rebase -i`` for commits squash, see `instructions <https://www.internalpointers.com/post/squash-commits-into-one-git>`_

* Push your commit from your feature branch. This will create a remote branch in your forked repository on GitHub, from which you will raise a PR.

.. code-block:: bash

  git push --set-upstream origin <name-of-change>

* Create a PR from the official TensorCircuit repository and send it for review. Some comments and remarks attached with the PR are recommended. If the PR is not finally finished, please add [WIP] at the beginning of the title of your PR.

* The PR will be reviewed by the developers and may get approved or change requested. In the latter case, you can further revise the PR according to suggestions and feedback from the code reviewers.

* The PR you opened can be automatically updated once you further push commits to your forked repository. Please remember to ping the code reviewers in the PR conversation soon.

* Please always include new docs and tests for your PR if possible and record your changes on CHANGELOG.


Checks
--------------------

The simplest way to ensure the codebase is ok with checks and tests is to run one-in-all scripts ``./check_all.sh`` (you may need to ``chmod +x check_all.sh`` to grant permissions on this file).

The scripts include the following components:

* black

* mypy: configure file is ``mypy.ini``, results strongly correlated with the version of numpy, we fix ``numpy==1.21.5`` as mypy standard in CI.

* pylint: configure file is ``.pylintrc``

* pytest: see :ref:`Pytest` sections for details. 

* sphinx doc builds: see :ref:`Docs` section for details.

Make sure the scripts check are successful by 💐.

Similar tests and checks are also available via GitHub action as CI infrastructures.

Please also include corresponding changes for CHANGELOG.md and docs for the PR.


Pytest
---------

For pytest, one can speed up the test by ``pip install pytest-xdist``, and then run parallelly as ``pytest -v -n [number of processes]``. 
We also have included some micro-benchmark tests, which work with ``pip install pytest-benchmark``.

**Fixtures:**

There are some pytest fixtures defined in the conftest file, which are for customization on backends and dtype in function level.
``highp`` is a fixture for complex128 simulation. While ``npb``, ``tfb``, ``jaxb`` and ``torchb`` are fixtures for global numpy, tensorflow, jax and pytorch backends, respectively.
To test different backends in one function, we need to use the parameterized fixture, which is enabled by ``pip install pytest-lazy-fixture``. Namely, we have the following approach to test different backends in one function.

.. code-block:: python

    from pytest_lazyfixture import lazy_fixture as lf

    @pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb"), lf("torchb")])
    def test_parameterized_backend(backend):
        print(tc.backend.name)



Docs
--------

We use `sphinx <https://www.sphinx-doc.org/en/master/>`__ to manage the documentation.

The source files for docs are .rst file in docs/source.

For English docs, ``sphinx-build source build/html`` and ``make latexpdf LATEXMKOPTS="-silent"`` in docs dir are enough.
The html and pdf version of the docs are in docs/build/html and docs/build/latex, respectively.

**Formula Environment Attention**

It should be noted that the formula environment ``$$CONTENT$$`` in markdown is equivalent to the ``equation`` environment in latex.
Therefore, in the jupyter notebook documents, do not nest the formula environment in ``$$CONTENT$$`` that is incompatible with
``equation`` in latex, such as ``eqnarray``, which will cause errors in the pdf file built by ``nbsphinx``.
However, compatible formula environments can be used. For example, this legal code in markdown

.. code-block:: markdown

    $$
    \begin{split}
        X&=Y\\
        &=Z
    \end{split}
    $$

will be convert to

.. code-block:: latex

    \begin{equation}
        \begin{split}
            X&=Y\\
            &=Z
        \end{split}
    \end{equation}

in latex automatically by ``nbsphinx``, which is a legal latex code. However, this legal code in markdown

.. code-block:: markdown

    $$
    \begin{eqnarray}
        X&=&Y\\
        &=&Z
    \end{eqnarray}
    $$

will be convert to

.. code-block:: latex

    \begin{equation}
        \begin{eqnarray}
            X&=&Y\\
            &=&Z
        \end{eqnarray}
    \end{equation}

in latex, which is an illegal latex code.

**Auto Generation of API Docs:**

We utilize a python script to generate/refresh all API docs rst files under /docs/source/api based on the codebase /tensorcircuit.

.. code-block:: bash

    cd docs/source
    python generate_rst.py

**i18n:**

For Chinese docs, we refer to the standard i18n workflow provided by sphinx, see `here <https://www.sphinx-doc.org/en/master/usage/advanced/intl.html>`__.

To update the po file from updated English rst files, using

.. code-block:: bash

    cd docs
    make gettext
    sphinx-intl update -p build/gettext -l zh


Edit these .po files to add translations (`poedit <https://poedit.net/>`__ recommended). These files are in docs/source/locale/zh/LC_MESSAGES.

To generate the Chinese version of the documentation: ``sphinx-build source -D language="zh" build/html_cn`` which is in the separate directory ``.../build/html_cn/index.html``, whereas English version is in the directory ``.../build/html/index.html``.


Releases
------------

Firstly, ensure that the version numbers in __init__.py and CHANGELOG are correctly updated.

**GitHub Release**

.. code-block:: bash

    git tag v0.x.y 
    git push origin v0.x.y
    # assume origin is the upstream name

And from GitHub page choose draft a release from tag.

**PyPI Release**

.. code-block:: bash

    python setup.py sdist bdist_wheel
    export VERSION=0.x.y
    twine upload dist/tensorcircuit-${VERSION}-py3-none-any.whl dist/tensorcircuit-${VERSION}.tar.gz


**DockerHub Release**

Make sure the DockerHub account is logged in via ``docker login``.

.. code-block:: bash

    sudo docker build . -f docker/Dockerfile -t tensorcircuit
    sudo docker tag tensorcircuit:latest tensorcircuit/tensorcircuit:0.x.y
    sudo docker push tensorcircuit/tensorcircuit:0.x.y
    sudo docker tag tensorcircuit:latest tensorcircuit/tensorcircuit:latest
    sudo docker push tensorcircuit/tensorcircuit:latest

**Binder Release**

One may need to update the tensorcirucit version for binder environment by pushing new commit in refraction-ray/tc-env repo with new version update in its ``requriements.txt``.
See `mybind setup <https://discourse.jupyter.org/t/tip-speed-up-binder-launches-by-pulling-github-content-in-a-binder-link-with-nbgitpuller/922>`_ for speed up via nbgitpuller. 