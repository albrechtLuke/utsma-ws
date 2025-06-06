## Plan to look at:
1. Install and run ROS2 on Jetson 
	* May require the new ssd before it actually works :(
	* once we have the new ssd I want to reformat the OS drive
2. Upgrading Vision Model
	* w/ ssd
	* make sure svo is recordable
	* make testing track w/ cones
	* record svo of track drive
	* split video into frames
	* use current model to draw & save bbox
	* 


### What we need from ZED
* Object detection bbox (preferably 3d at this stage)
	* [General ZED ROS2 Doc](https://www.stereolabs.com/docs/ros2/object-detection)
	* [Docs for Custom models](https://www.stereolabs.com/docs/ros2/custom-object-detection)
	* personally not sure about how the bbox becomes 3d, if it is accurate, to be trusted, etc. need to look into it further 
	* At first my plan for this was for it to be outside of the ROS stack due to latency concerns but it may actually be feasible and better inside the stack as long as we can get [IPC](https://docs.ros.org/en/foxy/Tutorials/Demos/Intra-Process-Communication.html) working well
* Depth Sensing
	* [ZED ROS2 Docs](https://www.stereolabs.com/docs/ros2/depth-sensing)


## Thoughts on SLAM
* [ROS2 slam comparison](https://github.com/Tanishq30052002/ros2-slam-comparison) might be interesting to look at 
* [Comparing Different SLAM Methods](https://adityakamath.github.io/2021-09-05-comparing-slam-methods/) also this is worth a read
* 



### Other notes 
* hehe itd be funny if you could drive the car with an xbox controller
