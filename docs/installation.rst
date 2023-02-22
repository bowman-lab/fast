installation
===============

Go to the directory where you would like to install fast and clone repo::

        cd /dir/to/install/packages
        git clone http://github.com/mizimmer90/fast.git

Create a conda environment for fast::

        conda create --name fast python=3.6

Within this conda environment, install dependencies (enspara, mdtraj, and their dependencies)::

        conda activate fast
        conda install -c conda-forge mdtraj=1.9.3
        conda install cython
        conda install -c confa-forge mpi4py

Follow directions for installing enspara [https://enspara.readthedocs.io/en/latest/installation.html]

Go to fast install location and run setup scripts::

        cd /dir/to/install/packages/fast
        python setup.py build
        python setup.py install
        

