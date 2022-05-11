Run TensorCircuit on Windows Machine with Docker
========================================================

Contributed by `SexyCarrots <https://github.com/SexyCarrots>`_ (Xinghan Yang)

(For linux machines, please review the `Docker README for linux <https://github.com/quclub/tensorcircuit-dev/blob/master/docker/README.md>`_ )

This note is only a step-by-step tutorial to help you build and run a Docker Container for Windows Machine users with the given dockerfile. 
If you want to have a deeper dive in to Docker, please check the official `Docker Orientation <https://docs.docker.com/get-started/>`_
and free courses on `YouTube <https://www.youtube.com/results?search_query=docker+tutorial>`_.

Why We Can't Run TensorCircuit on Windows Machine
---------------------------------------------------------------

Due to the compatability issue with the `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ backend on Windows,
we could not directly use jax backend for TensorCircuit on Windows machines. Please be aware that it is possible to `install
JAX on Windows <https://jax.readthedocs.io/en/latest/developer.html>`_, but it is tricky and not recommended unless
you have solid understanding of Windows environment and C++ tools. Virtual machine is also an option for development if
you are familiar with it. In this tutorial we would discuss the deployment of Docker for TensorCircuit since it use 
the most convenient and workable solution for beginners.

What Is Docker
------------------

Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly.
With Docker, you can manage your infrastructure in the same way as you manage your applications. By taking advantage of Docker's methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production.

(Source: https://docs.docker.com/get-started/overview/) 

For more information and tutorials on Docker, you could check the `Docker Documentation <https://docs.docker.com/get-started/overview/>`_.

Install Docker and Docker Desktop
---------------------------------------------

`Download Docker Desktop for Windows <https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe>`_ for and install it by following its instructions.

*Following information is from the official Docker Doc: https://docs.docker.com/desktop/windows/install/*

**Install interactively**

- If you haven't already downloaded the installer (Docker Desktop Installer.exe), you can get it from Docker Hub. It typically downloads to your Downloads folder, or you can run it from the recent downloads bar at the bottom of your web browser.

- When prompted, ensure the Use WSL 2 instead of Hyper-V option on the Configuration page is selected or not depending on your choice of backend.

- If your system only supports one of the two options, you will not be able to select which backend to use.

- Follow the instructions on the installation wizard to authorize the installer and proceed with the install.

- When the installation is successful, click Close to complete the installation process.

- If your admin account is different to your user account, you must add the user to the docker-users group.
Run Computer Management as an administrator and navigate to Local Users and Groups > Groups > docker-users. Right-click to add the user to the group. Log out and log back in for the changes to take effect.

**Install from the command line**

After downloading Docker Desktop Installer.exe, run the following command in a terminal to install Docker Desktop:

.. code-block:: bash

    "Docker Desktop Installer.exe" install

If you're using PowerShell you should run it as:

.. code-block:: bash

    Start-Process '.\win\build\Docker Desktop Installer.exe' -Wait install

If using the Windows Command Prompt:

.. code-block:: bash

    start /w "" "Docker Desktop Installer.exe" install

Build Image in through PyCharm or Command Line Interface
--------------------------------------------------------

**First of all**, run docker desktop.

**For CLI command:**

Go to your local ``./tensorcircuit-dev/docker`` directory, then open your local CLI.

.. code-block:: bash

    cd ./tensorcircuit-dev/docker

Use the command:

.. code-block:: bash

    docker build .

It could take more than fifteen minutes to build the docker image, depending on your internet and computer hardware.
Please keep your computer active while building the docker image. You need to build the image again from scratch if
there is any interruption during the building.

**For PyCharm:**

Install the docker plugin within Pycharm, than open the dockerfile in the ``./tensorcircuit-dev/docker`` directory.
Choose Dockerfile to be the configuration, then run the dockerfile.
Please keep your computer active while building the docker image. You need to build the image again from scratch if
there is any interruption during the building.

Run Docker Image and Examples in TensorCircuit
--------------------------------------------------------

Open your CLI

Find your local images by:

.. code-block:: bash

    docker images

Run image as a container by:

.. code-block:: bash

    docker run [image name]

List existing containers by:

.. code-block:: bash

    docker ps

Then, open docker desktop and open docker CLI:

.. code-block:: bash

    ls

You would see all files and directories in ``./tensorcircuit-dev/`` listed.

Go to the dir where all examples are:

.. code-block:: bash

    cd examples

Again, to see all the examples:

.. code-block:: bash

    ls

We would run noisy_qml.py to see what would happen:

.. code-block:: bash

    python noisy_qml.py

See the result and play with other example for a while. Latter you could start developing your own projects within
the docker container we just built. Enjoy your time with TensorCircuit.

*Please don't hesitate to create a New issue in GitHub if you find problems or have anything for discussion with other contributors*
