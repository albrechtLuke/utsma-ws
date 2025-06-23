import pyzed.sl as sl

def init_camera(resolution=sl.RESOLUTION.HD1200, depth_mode=sl.DEPTH_MODE.NEURAL):
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=resolution,
        depth_mode=depth_mode,
        coordinate_units=sl.UNIT.METER
    )
    zed.open(init_params)
    runtime = sl.RuntimeParameters()
    return zed, runtime

def grab_data(zed, runtime):
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()
        point_cloud = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        return image, point_cloud
    else:
        return None, None
    
def get_pose(zed):
    pose = sl.Pose()
    zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
    return pose

