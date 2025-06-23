import time
import json

import argparse
import cv2
import numpy as np

import pyzed.sl as sl

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer




def __main(opt: argparse.Namespace):
    print("Initializing Camera...")
    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)

    init_params.camera_resolution = sl.RESOLUTION.HD1200
    init_params.camera_fps = 60

    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL.PERFORMANCE

    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50
    is_playback = opt.svo is not None and len(opt.svo) > 0 # Defines if an SVO is used

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open : {repr(status)}. Exit program.")
        exit()
    camera_configuration = zed.get_camera_information().camera_configuration
    print("Initializing Camera... DONE")

    # Enable positional tracking module
    print("Enabling Positional Tracking...")
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static in space, enabling this setting below provides better depth quality and faster computation
    # positional_tracking_parameters.set_as_static = True
    status = zed.enable_positional_tracking(positional_tracking_parameters)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Positional Tracking enable : {repr(status)}. Exit program.")
        zed.close()
        exit()
    print("Enabling Positional Tracking... DONE")

    # Enable object detection module
    print("Enabling Object Detection...")

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
    obj_param.custom_onnx_file = opt.custom_onnx
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = True  # designed to give person pixel mask using Stereolabs internal models
    # if opt.custom_labels:
    #     obj_param.custom_model_label = [name.strip() for name in opt.custom_labels.split(',')]


    status = zed.enable_object_detection(obj_param)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Object Detection enable : {repr(status)}. Exit program.")
        zed.close()
        exit()
    print("Enabling Object Detection... DONE")

    # Setting custom OD runtime parameters
    detection_parameters_rt = sl.CustomObjectDetectionRuntimeParameters()
    # Default properties, apply to all object class
    detection_parameters_rt.object_detection_properties.detection_confidence_threshold = 30
    # Specific properties, override the default properties
    props_dict = {
        1: sl.CustomObjectDetectionProperties(),
        2: sl.CustomObjectDetectionProperties()
    }
    props_dict[1].native_mapped_class = sl.OBJECT_SUBCLASS.PERSON
    props_dict[1].object_acceleration_preset = sl.OBJECT_ACCELERATION_PRESET.MEDIUM
    props_dict[1].detection_confidence_threshold = 40
    props_dict[2].detection_confidence_threshold = 50
    props_dict[2].max_allowed_acceleration = 10 * 10
    detection_parameters_rt.object_class_detection_properties = props_dict

    quit_bool = False
    if not opt.disable_gui:
        image_aspect_ratio = camera_configuration.resolution.width/camera_configuration.resolution.height
        requested_low_res_w = min(1280, camera_configuration.resolution.width)

        display_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w/image_aspect_ratio)
        image_scale = [display_resolution.width / camera_configuration.resolution.width,
                       display_resolution.height / camera_configuration.resolution.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239,255], np.uint8)

        # Utilities for tracks view
        camera_config = zed.get_camera_information().camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance*1000, 2, opt.custom_labels)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

        # Will store the 2D image and tracklet views 
        global_image = np.full((display_resolution.height, display_resolution.width+tracks_resolution.width, 4), [245, 239, 239,255], np.uint8)
        viewer = gl.GLViewer()
        pc_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio) 
        viewer.init(zed.get_camera_information().camera_model, pc_resolution, obj_param.enable_tracking)
        point_cloud = sl.Mat(pc_resolution.width, pc_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        objects = sl.Objects()
        image_left = sl.Mat()
        cam_w_pose = sl.Pose()
        image_scale = (display_resolution.width/camera_config.resolution.width,display_resolution.height/camera_config.resolution.height)

    objects = sl.Objects() 
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    window_name = "ZED | 3D View tracking"
    gl_viewer_available = True

    latency_list = []
    latency_list_first = True

    __printHelp()

    while 1:

        if not opt.disable_gui and (zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS or quit_bool):
            break
        if opt.disable_gui and (zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS or quit_bool or not gl_viewer_available):
            break


        detect_start = time.perf_counter()

        status = zed.retrieve_custom_objects(objects, detection_parameters_rt)

        detect_end = time.perf_counter()
        latency_ms = (detect_end - detect_start) * 1000.0
        latency_list.append(latency_ms)
        if len(latency_list) > 500:
            if latency_list_first:
                batch = latency_list[10:]
                avg_latency = sum(batch) / len(batch)
                latency_list = [avg_latency]
                print(f"!!![Average] {avg_latency:.2f} ms")
                latency_list_first = False
            else:
                avg_latency = sum(latency_list) / len(latency_list)
                print(f"[Average] {avg_latency:.2f} ms")
                latency_list = [avg_latency]

        print(f"[Latency] {latency_ms:.2f} ms")
        


        if status == sl.ERROR_CODE.SUCCESS:
            if not opt.disable_gui:

                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, pc_resolution)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_render_left = image_left.get_data()
                np.copyto(image_left_ocv,image_render_left)
                track_view_generator.generate_view(objects, image_left_ocv,image_scale ,cam_w_pose, image_track_ocv, objects.is_tracked)
                global_image = cv2.hconcat([image_left_ocv,image_track_ocv])
                viewer.updateData(point_cloud, objects)

                gl_viewer_available = viewer.is_available()


                detections = []


                cv2.imshow(window_name, global_image)
                key = cv2.waitKey(10)
                if key == ord('q'):
                    quit_bool = True
                if key == ord('i'):
                    track_view_generator.zoomIn()
                if key == ord('o'):
                    track_view_generator.zoomOut()

        if is_playback and zed.get_svo_position() == zed.get_svo_number_of_frames() - 1:
            quit_bool = True


    if not opt.disable_gui:
        viewer.exit()
        point_cloud.free()
        image_left.free()
    if len(latency_list) > 10:
        avg_latency = sum(latency_list) / len(latency_list)
        print(f"[Summary] {avg_latency:.2f} ms")
    else:
        print("\n[Summary] Not enough measurements")
    zed.disable_object_detection()
    zed.close()


def __printHelp():
    print("\n\n Birds eye view hotkeys:")
    print("* Zoom in tracking view            'i'")
    print("* Zoom out tracking view           'o'")
    print("* Exit:                            'q'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_onnx', type=str, required=True, help='Path to custom ONNX model to use')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--disable_gui', action = 'store_true', help='Flag to disable the GUI to increase detection performances. On low-end hardware such as Jetson Nano, the GUI significantly slows down the detection and increase the memory consumption')
    parser.add_argument('--custom_labels', type=str, default="", help="Comma-seperated list of class names (e.g. 'blue_cone,yellow_cone')")
    opt = parser.parse_args()

    __main(opt)
