import collections
import os
import sys
import time
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from openvino.inference_engine import IECore
from decoder import OpenPoseDecoder
from sort import sort

# A directory where the model will be downloaded.
base_model_dir = "models"

# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"

# Selected precision (FP32, FP16, FP16-INT8).
precision = "F16"

model_path = f"models/{precision}/{model_name}.xml"
model_weights_path = f"models/{precision}/{model_name}.bin"

# Initialize OpenVINO Runtime
ie_core = IECore()

# Read the network and corresponding weights from a file.
net = ie_core.read_network(model_path, weights=model_weights_path)
compiled_model = ie_core.load_network(network=net, device_name="MYRIAD")

# print(compiled_model.input_info)
# print(compiled_model.outputs['Mconv7_stage2_L1'])
# input()

input_layer = next(iter(net.input_info))
#print(input_layer)

n, c, height, width = net.input_info[input_layer].input_data.shape
#print(n, c, height, width)

# Get the input and output names of nodes.
# input_layer = compiled_model.input(0)
output_layers = list(compiled_model.outputs)

decoder = OpenPoseDecoder()
# initialize sorting algo 
sort_tracker = sort.Sort(max_age=15,min_hits=2, iou_threshold=0.01) 

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
    bboxes = []
    for pose in poses:
        offset = 10
        xmin = 1000.0
        xmax = 1.0
        ymin = 1000.0
        ymax = 1.0
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
                _x = int(p[0])
                _y = int(p[1])
                if _x < xmin:
                    xmin = _x
                if _x > xmax:
                    xmax = _x
                    
                if _y < ymin:
                    ymin = _y
                if _y > ymax:
                    ymax = _y
        xmin = xmin - (offset+offset) 
        ymin = ymin - (offset+offset) 
        xmax = xmax + offset 
        ymax = ymax + (offset+offset) 
        if xmin < 0:
            xmin = 1
        if ymin < 0:
            ymin = 1
        bboxes.append((xmin,ymin,xmax,ymax))
        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,21,210),1)
            
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    #print(f" \n\n\n length of poses is {i} \n\n\n\n")
    return {"Image":img, "Boxes":bboxes}

# Main processing function to run pose estimation.
def run_pose_estimation(source=0, flip=False, use_popup=True, skip_first_frames=0):
    pafs_output_key =  output_layers[0] #compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = output_layers[1] # compiled_model.output("Mconv7_stage2_L2")

    player = None
    try:
        # Create a video player to play with target fps.
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

            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Resize the image and change dims to fit neural network input.
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2,0,1))[np.newaxis, ...]

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
            # Get poses from network results.
            poses, scores = process_results(frame, pafs, heatmaps)

            # Draw poses on a frame.
            obj = draw_poses(frame, poses, 0.1)
            try:
                frame = obj["Image"]
                bboxes = obj["Boxes"]
                print(bboxes)
                dets_to_sort = np.empty((0,6))
                for box in bboxes:
                    xmin, ymin, xmax, ymax = box
                    dets_to_sort = np.vstack((dets_to_sort, np.array([xmin, ymin, xmax, ymax, 0.9, 15.0])))
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

            except Exception as error:
                pass
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

            #frame = cv2.resize(frame,(1920,1080))
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
