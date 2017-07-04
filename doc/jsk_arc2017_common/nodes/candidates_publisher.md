# candidates\_publisher.py


## What is this?

Publish label candidates from JSON file.


## Subscribing topics


- `~input/json_dir` (`std_msgs/String`)

  JSON file directory

## Publishing topics

- `~output/candidates` (`jsk_recognition_msgs/LabelArray`)

  Label candidates in target location


## Parameters

- `~target_location` (String, required)

   Target location name.
   (`tote`, `bin_A`, `bin_B` or `bin_C`)

   You can update by `dynamic_reconfigure`.

- `~label_names` (List of String, required)

  List of label names
  
## Sample

```bash
roslaunch jsk_arc2017_common sample_candidates_publisher.launch
```
