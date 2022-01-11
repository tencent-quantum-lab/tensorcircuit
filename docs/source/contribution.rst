Guide for contributors
============================

We welcome everyone’s contributions. The development of tensorcircuit are open-sourced and centered on GitHub.

There are various ways to contribute:

* Answering questions on tensorcircuit’s discussions page or issue page.

* Rasing issues such as bug report or feature request on tensorcircuit's issue page.

* Improving tensorcircuit's documentation (docstrings/tutorials) by pull requests.

* Contributing to tensorcircuit's codebase by pull requests.



Pull Request Guidelines
-------------------------------

We welcome pull requests from everyone. But for large PRs related to feature enhencement or API changes, we ask that you first open a GitHub issue to discuss on your proposals.

We develop tensorcircuit using git on GitHub, so basic knowledge on git and GitHub is assumed.

The following git workflow is recommended to contribute by PR:

* Fork the tensorcircuit GitHub repository by clicking the Fork button from GitHub. This will create a fork version of the code repository in your own GitHub account.

* Configure the python environment locally for development. ``pip install -r requirements.txt`` and ``pip install -r requirements-dev.txt`` are recommended. Extra packages may be required for specific development tasks.

* Clone your fork repository locally and setup upstreams to the official tensorcircuit repository. And configure your git user and email so that they match your GitHub account if you haven't.

.. code-block:: bash

    git clone <your-forked-repo-git-link>
    cd tensorcircuit
    git remote add upstream <offical-repo-git-link>
    git config user.name <GitHub name>
    git config user.email <GitHub email>

* Pip installing your fork from source. This allows you to modify the code locally and immediately test it out, ``python setup.py develop``

* Create a feature branch where you will develop from, don't open PR from your master/main branch, ``git checkout -b name-of-change``.

* Make sure the changes can pass all checks by running: ``./check_all.sh``. (See details for checks below)

* Once you are satisfied with your change, create a commit as follows:

.. code-block:: bash

    git add file1.py file2.py ...
    git commit -m "Your commit message (should be brief and informative)"
    
* You should sync your code with the official repo:

.. code-block:: bash

    git fetch upstream
    git rebase upstream/master
    # resolve conflicts if any

* Note that PRs typically comprise a single git commit, you should squash all you commits in the feature branch. Using ``git rebase -i`` for commits squash, see `instructions <https://www.internalpointers.com/post/squash-commits-into-one-git>`_

* Finally, push your commit from your feature branch and create a remote branch in your forked repository on GitHub that you can use to create a pull request from: ``git push --set-upstream origin name-of-change``.

* Create a pull request from official tensorcircuit repository and send it for review. Some comments and remarks attached with the PR are recommended. If the PR is not finally finished, please add [WIP] at the begining of the title of your PR.

* The PRs will be reviewed by developers of tensorcircuit and it will get approved or change requested. In the latter case, you can further revise the PR according to suggestions and feedbacks from the code reviewers.

* The PRs you opened can be automatically updated once you further push commits to your forked repository. Please ping the code reviewer in the PR dialogue once you finished the change.

* Please always include new docs and tests for your PR if possible and record your changes on CHANGELOG.


Checks and Tests
--------------------

The simplest way to ensure the codebase are ok with checks and tests, is to run one-in-all scripts ``./check_all.sh`` (you may need to ``chmod +x check_all.sh`` to grant permissions on this file).

The scripts include the following components:

* black

* mypy

* pylint

* pytest: For pytest, one can speed up the test by ``pip install pytest-xdist``, and then run parallelly as ``pytest -v -n [number of processes]``.

* sphinx doc builds

Make sure the scripts check are successful by 💐.

The similar tests and checks are also available via GitHub action as CI infrastructures.

Please also include corresponding changes for CHANGELOG.md and docs for the PR.


Docs
--------

We use `sphnix <https://www.sphinx-doc.org/en/master/>`__ to manage the documentations.

The source files for docs are .rst file in docs/source.

For English docs, ``make html`` in docs dir is enough. The html version of the docs are in docs/build/html.

**i18n:**

For Chinese docs, we refer to standard i18n workflow provided by sphnix, see `here <https://www.sphinx-doc.org/en/master/usage/advanced/intl.html>`__.

To update the po file from updated English rst files, using

.. code-block:: bash

    cd docs
    make gettext
    sphinx-intl update -p build/gettext -l cn


Edit these .po files to add translations (`poedit <https://poedit.net/>`__ recommended). These files are in docs/source/locale/cn/LC_MESSAGES.

Generate Chinese version of the documentation: ``make -e SPHINXOPTS="-D language='cn'" html``.