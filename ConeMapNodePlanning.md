cap
## what this will have:
* subscriber to object detection topic
* publish map as markerArray, (ie. visualization_msgs/MarkerArray)
* publish map as custom message
* `tf2` buffer and listener to transform cone positions to map frame
* store cones somewhere (eg. dict where key is trackingID)
* use trackingID to update existing cones, else use pos and distance threshold to update existing/new cones
* publish map every time theres a new detection, published map at fixed rate

## steps
* for each object:
	* extract 3d pos (pos is given in camera frame, transform to map frame)
	* extract trackingID and dimensions
* transform position to map frame, use `tf2`
* if tracking available:
	* yes: check for that ID on map, if yes update position and reset age, if no create new cone. 
	* no: check for cones within threshold and update cone closest, or create new cone. 
* publish the updated map as markers, sync or async tbd


## thoughts
* if trackingID is not reliable we might be better off using clustering algs (ie. DBSCAN) to merge detections
* 