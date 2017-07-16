# annotate_2d_dataset_20170714

## Usage

Please see this. @YutoUchimi @708yamaguchi

```bash
sudo pip install gshell

# @708yamaguchi
gshell download --with-id 0B9P1L--7Wd2vYzdfX3YtcTMwMm8
unzip dataset_jsk_v3_20160614_yamaguchi.zip
rosrun jsk_arc2017_common annotate_2d_dataset.py -d dataset_jsk_v3_20160614_yamaguchi

# @YutoUchimi
gshell download --with-id 0B9P1L--7Wd2vMkNGWEV1LU9ZTDg
unzip dataset_jsk_v3_20160614_uchimi.zip
rosrun jsk_arc2017_common annotate_2d_dataset.py -d dataset_jsk_v3_20160614_uchimi
```

## Memo

Shared data: https://drive.google.com/open?id=0B9P1L--7Wd2vYmVMeXFpbWdINE0
