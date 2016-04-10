euslisp/jsk_2015_05_baxter_apc/baxter-interface.l
=================================================

jsk\_2015\_05\_baxter\_apc::baxter-interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  :super **baxter-interface**
-  :slots *tfl *\ bin-boxes *objects-in-bin-boxes *\ objects-in-bin-coms
   \_bin-coords-list

:wait-for-user-input-to-start *arm*

:init *&rest* *args*

:start-grasp *&optional* *(arm :arms)*

:stop-grasp *&optional* *(arm :arms)*

:graspingp *arm*

:opposite-arm *arm*

:need-to-wait-opposite-arm *arm*

:arm-symbol2str *arm*

:arm-potentio-vector *arm*

:tf-pose->coords *frame\_id* *pose*

:fold-pose-back *&optional* *(arm :arms)*

:detect-target-object-in-bin *target-object* *bin*

:recognize-bin-boxes *&key* *(stamp (ros::time-now))*

:bbox->cube *bbox*

:visualize-bins

:visualize-objects

:recognize-grasp-coords-list *bin* *&key* *(stamp (ros::time-now))*

:recognize-objects-in-bin *bin* *&key* *(stamp (ros::time-now))*
*(timeout 10)*

:recognize-object-in-hand *arm* *&key* *(stamp (ros::time-now))*
*(timeout)*

:verify-object *arm* *object-name* *&key* *(stamp (ros::time-now))*

:try-to-pick-in-bin *arm* *bin*

:try-to-pick-object-solidity *arm* *bin* *&key* *(offset #f(0.0 0.0
0.0))*

:try-to-pick-object *arm* *bin* *&key* *(object-index 0)* *(offset
#f(0.0 0.0 0.0))*

:pick-object *arm* *bin* *&key* *(object-index 0)* *(n-trial 1)*
*(n-trial-same-pos 1)* *(do-stop-grasp nil)*

:send-av *&optional* *(tm 3000)*

:force-to-reach-goal *&key* *(arm :arms)* *(threshold 5)* *(stop 10)*

:ik->bin-entrance *arm* *bin* *&key* *(offset #f(0.0 0.0 0.0))*

:move-arm-body->bin *arm* *bin*

:move-arm-body->order-bin *arm*

:spin-off-by-wrist *arm* *&key* *(times 10)*

:move-arm-body->head-view-point *arm*

:place-object *arm*

:get-work-orders *arm*

:get-next-work-order *arm* *current-order*

:get-bin-contents *bin*

:real-sim-end-coords-diff *arm*

jsk\_2015\_05\_baxter\_apc::baxter-init *&key* *(ctype
:default-controller)*
