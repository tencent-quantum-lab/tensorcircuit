================
Quick Start
================

Install from GitHub
--------------------------

For beta version usage, one needs to install tensorcircuit package from GitHub. For development and PR workflow, please refer to `contribution <contribution.html>`__ instead.

For private tensorcircuit-dev repo, one needs to firstly configure the SSH key on GitHub and locally, please refer to `GitHub doc <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`__

Then try ``pip3 install --force-reinstall git+ssh://git@github.com/quclub/tensorcircuit-dev.git`` in shell.

Depending on one's need, one may further pip install tensorflow (for tensorflow backend) or jax and jaxlib (for jax backend) or `cotengra <https://github.com/jcmgray/cotengra>`__ (for more advanced tensornetwork contraction path solver).

