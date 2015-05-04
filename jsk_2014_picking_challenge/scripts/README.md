About object data
=================
To extract features from object data,
you need to download distributed by Robot Learning Lab, UC Berkeley.
Object data is available from here::

    * http://rll.berkeley.edu/amazon_picking_challenge/


Attention
---------
You should change dirname for following items manually::

    * kygen_squeakin_eggs_plush_puppies  -> kyjen_squeakin_eggs_plush_puppies
    * rollodex_mesh_collection_jumbo_pencil_cup -> rolodex_jumbo_pencil_cup


Extract sift
============
1. Follow the instruction at **About object data**
2. Execute following::

    $ roslaunch jsk_2014_picking_challenge extract_sift.launch


Kinect2で収集したデータの取得方法
=================================
```sh
roscd jsk_data
make large KEYWORD=jsk_2014_picking_challenge/20150428_collected_images.tgz
```


スコアの計算
============
```sh
rosrun jsk_2014_picking_challenge score_calculator.py
```
