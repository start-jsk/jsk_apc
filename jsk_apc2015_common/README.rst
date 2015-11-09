==================
jsk_apc2015_common
==================


Train classifier with dataset
=============================


Get dataset
-----------
To get dataset at `here <http://rll.berkeley.edu/amazon_picking_challenge/>`_, and run::

  python scripts/download_dataset.py -O berkeley_dataset


Process dataset
---------------
Firstly, you need to create mask applied image dataset::

  python scripts/create_mask_applied_dataset.py berkeley_dataset -O berkeley_dataset_mask_applied


Gazebo models
=============

The mesh files under ``models/`` are originally created by Arjun Singh, Karthik Narayan,
Ben Kehoe, Sachin Patil, Ken Goldberg, Pieter Abbeel in Robot Learning Lab, UC Berkeley.
Their website is `here <http://rll.berkeley.edu/amazon_picking_challenge>`_.
