# json\_saver.py


## What is this?

Update and save item location JSON file during the tasks.

## Services

- `~update_json` (`jsk_arc2017_common/UpdateJSON`)

  Update and save `item_location_file.json`

```bash
$ rossrv show jsk_arc2017_common/UpdateJSON
string item
string src
string dst
---
bool updated
```

- `~save_json` (`std_srvs/Trigger`)

  Save `item_location_file.json`.

## Parameters

- `~json_dir` (String, required)

  Directory where initial json files are located

- `~output_dir` (String, required)

  Directory where output json files will be located


## Sample

```bash
roslaunch jsk_arc2017_common sample_json_saver.launch
```
