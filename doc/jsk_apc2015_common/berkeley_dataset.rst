Setup Berkeley Dataset for Object Recognition
=============================================


Get dataset
-----------
To get dataset at `here <http://rll.berkeley.edu/amazon_picking_challenge/>`_::

  python scripts/download_dataset.py -O berkeley_dataset


Process dataset
---------------

To get mask applied images::

  python scripts/create_mask_applied_dataset.py berkeley_dataset -O berkeley_dataset_mask_applied

