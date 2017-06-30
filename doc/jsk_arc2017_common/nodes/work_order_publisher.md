# work\_order\_publisher.py

## What is this?

Publish optimized work orders of the tasks.


## Subscribing topics

None

## Publishing topics

- `~left_hand` (`jsk_arc2017_common/WorkOrderArray`)

  Optimized work orders for left hand.

- `~right_hand` (`jsk_arc2017_common/WorkOrderArray`)

  Optimized work orders for right hand.


## Parameters

- `~json_dir` (String, required)

  Directory where initial json files are located

- `~rate` (Int, default: `1`)

  Hz of publishing topics

## Sample

```bash
roslaunch jsk_arc2017_common sample_work_order_publisher.launch
```
