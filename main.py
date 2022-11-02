import collections
import os
import sys
import time
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from openvino.inference_engine import IECore


# import local classes/scripts 
from decoder import OpenPoseDecoder
from sort import sort


# A directory where the model will be downloaded.
base_model_dir = "models"

# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"
hmodel_name = "mobilenet-ssd"

# Selected precision (FP32, FP16, FP16-INT8).
precision = "F16"
hprecision = "ssdv3_model/FP16"

model_path = f"models/{precision}/{model_name}.xml"
model_weights_path = f"models/{precision}/{model_name}.bin"

hmodel_path = f"{hprecision}/{hmodel_name}.xml"
hmodel_weights_path = f"{hprecision}/{hmodel_name}.bin"


# initialize sorting algo 
sort_tracker = sort.Sort(max_age=15,min_hits=2, iou_threshold=0.1) 
# Initialize OpenVINO Runtime
ie_core = IECore()

# Read the network and corresponding weights from a file.
net = ie_core.read_network(model_path, weights=model_weights_path)
hnet = ie_core.read_network(hmodel_path,  weights=hmodel_weights_path)
compiled_model = ie_core.load_network(network=net, device_name="MYRIAD")
exec_net = ie_core.load_network(network=hnet, device_name="MYRIAD")


input_layer = next(iter(net.input_info))
input_blob = next(iter(hnet.input_info))
#hnet.input_info[input_blob].precision = 'U8'
output_blob = next(iter(hnet.outputs))
"""
if len(net.outputs) == 1:
    output_blob = next(iter(hnet.outputs))
    hnet.outputs[output_blob].precision = 'FP32'
else:
    hnet.outputs['boxes'].precision = 'FP32'
    hnet.outputs['labels'].precision = 'U16'
"""
n, c, height, width = net.input_info[input_layer].input_data.shape
_, _, net_h, net_w = hnet.input_info[input_blob].input_data.shape
#print(n, c, height, width)

# Get the input and output names of nodes.
# input_layer = compiled_model.input(0)
output_layers = list(compiled_model.outputs)

decoder = OpenPoseDecoder()

def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results(img, pafs, heatmaps):
    # This processing comes from
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)

    output_shape = list(compiled_model.outputs['Mconv7_stage2_L1'].shape)

    output_scale = img.shape[1] / output_shape[3], img.shape[0] / output_shape[2]
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores

colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))


def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
               
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
                
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return img

# Main processing function to run pose estimation.
def run_pose_estimation(source=0, flip=False, use_popup=True, skip_first_frames=0):
    pafs_output_key =  output_layers[0] #compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = output_layers[1] # compiled_model.output("Mconv7_stage2_L2")

    player = None
    try:
        # Create a video player to play with target fps.
        #player = cv2.VideoCapture('4.mp4')
        player = cv2.VideoCapture(0)
        ret = player.set(3,720)
        ret = player.set(4,480)
        
        processing_times = collections.deque()
        frame_number = 1
        
        

        while True:

            t1 = time.time()
            frame_number += 1
            # Grab the frame.
            ret,frame = player.read()
            if frame is None:
                print("Source ended")
                break
            
            # copy frame for ssd detection and tracking / copy image size : 300x300
            #himage = frame.copy()
            #if himage.shape[:-1] != (net_h, net_w):
            #    #log.warning(f'Image {args.input} is resized from {image.shape[:-1]} to {(net_h, net_w)}')
            himage = cv2.resize(frame, (net_w, net_h),interpolation=cv2.INTER_AREA)

            

            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Resize the image and change dims to fit neural network input.
            
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2,0,1))[np.newaxis, ...]
            
            # Change data layout from HWC to CHW
            himage = himage.transpose((2, 0, 1))
            # Add N dimension to transform to NCHW
            himage = np.expand_dims(himage, axis=0)

            # Measure processing time.
            start_time = time.time()

            # print("TIME0 : ", start_time - t1)
            

            if (frame_number % 2 == 0):
                # Get results.
                results = compiled_model.infer({'data': input_img})

            stop_time = time.time()

            # print("inference TIME: ", stop_time - start_time)

            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            
            # get detection result 
            res = exec_net.infer(inputs={input_blob: himage})
            
            # Get poses from network results.
            poses, scores = process_results(frame, pafs, heatmaps)

            # Draw poses on a frame.
            frame = draw_poses(frame, poses, 0.1)
            w, h = frame.shape[1], frame.shape[0]
            #------------------  Tracker ------------------------
            #frame = cv2.resize(frame,(300,300), interpolation=cv2.INTER_AREA)
            if len(hnet.outputs) == 1:
                res = res[output_blob]
                # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
                # where N is the number of detected bounding boxes
                detections = res.reshape(-1,7)
            else:
                detections = res['boxes']
                #labels = res['labels']
                # Redefine scale coefficients
                w, h = w / net_w, h / net_h
            
            #print(f"Detections : {detections}")
            dets_to_sort = np.empty((0,6))
            
            for i, detection in enumerate(detections):
                
                if len(hnet.outputs) == 1:
                    _, class_id, confidence, xmin, ymin, xmax, ymax = detection
                else:
                    class_id = i
                    xmin, ymin, xmax, ymax, confidence = detection

                if confidence > 0.1 and class_id == 15.0: # class_id ==1.0 is human/person
                    label = class_id

                    xmin = int(xmin * w)
                    ymin = int(ymin * h)
                    xmax = int(xmax * w)
                    ymax = int(ymax * h)
                    
                    if ((xmin < 3.0) and (ymin < 3.0 ) and (xmax < 6.0) and (ymax < 6.0)):
                        # leave this 
                        pass
                    else:

                        print(f'coords = ({xmin}, {ymin}), ({xmax}, {ymax})')
                        dets_to_sort = np.vstack((dets_to_sort, np.array([xmin, ymin, xmax, ymax, confidence, class_id])))
            
            if dets_to_sort[0][0] < 3.0:
                dets_to_sort = np.empty((0,6))
                # wrong data 
                pass
            else:     
                print(f"\n\n\n\n {dets_to_sort}\n\n\n")       
                tracked_dets = sort_tracker.update(dets_to_sort)
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    #categories = tracked_dets[:, 4]
                    for i ,box in enumerate(bbox_xyxy):
                    
                        x1, y1, x2, y2 = [int(i) for i in box]
                        _w = x1 + x2
                        _h = y1 + y2
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        # box text and bar
                        #cat = int(categories[i]) if categories is not None else 0
                        id = int(identities[i]) if identities is not None else 0
                        # Draw a bounding box on a output image
                        obj = {"rect":(x1,y1,x2,y2), "id":id}
                        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID{id}", (x1+10, y1+10),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000

            fps = 1000 / processing_time
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)


            # Use this workaround if there is flickering.
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

            #frame = cv2.resize(frame,(400,180))
            cv2.imshow(title, frame)

            #print("Inference Time : " + str(processing_time) + ", " +  str(fps))

            tt2 = time.time()
            #print("time 2: ", tt2 - stop_time)


            t2 = time.time()
            #print("Total processing time for this frame: ", t2-t1)

            key = cv2.waitKey(1)
            # escape = 27
            if key == 27:
                break

    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pose_estimation(source=0, flip=True, use_popup=True)
