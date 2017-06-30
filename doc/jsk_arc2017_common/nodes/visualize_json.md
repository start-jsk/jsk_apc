# visualize\_json.py


## What is this?

Visualizes `item_location_file.json` and `order_file.json`.

**Item Location**
![](https://user-images.githubusercontent.com/4310419/27720914-d5e07802-5d97-11e7-881e-9ee2ebd2c888.png)

**Order**
![](https://user-images.githubusercontent.com/4310419/27720897-c5344718-5d97-11e7-9e50-fbcbbd622f47.png)


## Subscribing topics

None


## Publshing topics

- `~output/item_location_viz` (`sensor_msgs/Image`)

  Visualization of `item_location_file.json`.
  This is enabled if `item_location` is inside of `~types` of rosparam,
  and `~json_dir/item_location_file.json` is read.

- `~output/order_viz` (`sensor_msgs/Image`)

  Visualization of `order_file.json`.
  This is enabled if `order` is inside of `~types` of rosparam,
  and `~json_dir/order_file.json` is read.


## Parameters

- `~rate` (Int, default: `1`)

  Hz of publishing topics.

- `~types` (List of string, required)

  `item_location` or/and `order`.

- `~json_dir` (String, required)

  Directory location where json files are be located.


## Sample

```bash
roslaunch jsk_arc2017_common sample_visualize_json.launch
```
