# test_hand_eye_coordination

![](https://user-images.githubusercontent.com/4310419/28137956-0ec55ab0-678a-11e7-9b0c-588ddeb37161.jpg)

## Usage

```bash
$ roslaunch jsk_arc2017_baxter baxter.launch
$ roslaunch jsk_arc2017_baxter test_hand_eye_coordination.launch
$ mv ~/.ros/jsk_arc2017_baxter/test_hand_eye_coordination/*.csv \
  $(rospack find jsk_arc2017_baxter)/experiments/test_hand_eye_coordination/
```


```
$ python estimate_error.py  # reads left.csv, right.csv
>>>>>>>>>>>>>>>>>>>> /home/wkentaro/Projects/jsk_apc/src/start-jsk/jsk_apc/jsk_arc2017_baxter/experiments/test_hand_eye_coordination/left.csv <<<<<<<<<<<<<<<<<<<<
------------------------------ Average ------------------------------
N fused: 793
Matrix:
[[ 0.03942553  0.9992016   0.00646474  0.55941472]
 [ 0.99853416 -0.03915744 -0.03736621 -0.09818186]
 [-0.03708323  0.00792844 -0.99928073 -0.28060799]
 [ 0.          0.          0.          1.        ]]
Position: [ 0.55941472 -0.09818186 -0.28060799]
Position (simple): [ 0.54347867 -0.08095304 -0.27987296]
Orientation: (3.1336586700713127, 0.037091737603843405, 1.531333415048344)
---------------------- Std Deviation (Variance) ----------------------
x: 0.000305 (0.017467)
x (simple): 0.00754488481318
y: 0.001171 (0.034222)
y (simple): 0.0124412020665
z: 0.000514 (0.022677)
z (simple): 0.00923224818815
angle: 0.006254 (0.079085)
>>>>>>>>>>>>>>>>>>>> /home/wkentaro/Projects/jsk_apc/src/start-jsk/jsk_apc/jsk_arc2017_baxter/experiments/test_hand_eye_coordination/right.csv <<<<<<<<<<<<<<<<<<<<
------------------------------ Average ------------------------------
N fused: 385
Matrix:
[[ 0.03510174  0.99866837  0.03780671  0.52425135]
 [ 0.99931877 -0.0355057   0.01006669 -0.11773387]
 [ 0.01139564  0.03742759 -0.99923436 -0.32538503]
 [ 0.          0.          0.          1.        ]]
Position: [ 0.52425135 -0.11773387 -0.32538503]
Position (simple): [ 0.52851077 -0.08534123 -0.32894418]
Orientation: (3.104153884581372, -0.011395887079444732, 1.5356850930069414)
---------------------- Std Deviation (Variance) ----------------------
x: 0.000069 (0.008281)
x (simple): 0.00701615529075
y: 0.001199 (0.034632)
y (simple): 0.00990600962476
z: 0.000059 (0.007692)
z (simple): 0.00342719966754
angle: 0.000308 (0.017551)
```
