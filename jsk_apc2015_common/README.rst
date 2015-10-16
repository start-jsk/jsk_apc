===================
jsk_2015_apc_common
===================


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
