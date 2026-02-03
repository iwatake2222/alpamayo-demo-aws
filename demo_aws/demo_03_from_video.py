# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import cv2
import numpy as np
import os
import sys
import time
import torch
from collections import deque

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

from utility import (
    put_text_with_bg,
    draw_trajectories,
    draw_trajectory_projected,
    load_images_opencv,
    opencv_images_to_torch,
    create_dummy_ego_history
)


WIDTH_TRAJECTORY_WINDOW_PX = 200
MODEL_INPUT_NUM_FRAMES = 4
MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 384


print("Loading Alpamayo model...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)


ego_history_xyz, ego_history_rot = create_dummy_ego_history()

print("Opening input/output video...")
video_path = "demo_aws/sample_input.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Failed to open video file")

### Prepare Output
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read first frame from video")

out_path = "output.mp4"
fps = 30
input_w, input_h = frame.shape[1] + WIDTH_TRAJECTORY_WINDOW_PX, frame.shape[0]
writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (input_w, input_h)
)


cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)  # skip first few frames
frame_buffer = deque(maxlen=MODEL_INPUT_NUM_FRAMES)
frame_id = 0
while True:
    print(f"Processing frame {frame_id}...")
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id == 10 * 1000:
        break

    if frame_id % 10 != 0:
        continue


    frame_buffer.append(frame)
    current_input_image = frame.copy()

    print("Preparing input...")
    input_images = opencv_images_to_torch(list(frame_buffer))
    input_images = torch.nn.functional.interpolate(
        input_images,
        size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    messages = helper.create_message(input_images)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    print("Running inference...")
    # torch.cuda.manual_seed_all(42)
    start = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
            # max_generation_length=256,
            max_generation_length=64,
            return_extra=True,
        )
    end = time.time()
    print(f"Inference done. elapsed: {end - start:.3f} sec")

    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    print(extra)
    # print(pred_xy)
    reason_text_list = extra["cot"][0][0]

    traj_x = []
    traj_y = []
    for traj_i in range(pred_xy.shape[0]):
        y = pred_xy[traj_i, 0]  # (N,)
        x = pred_xy[traj_i, 1]  # (N,)
        traj_x.append(-x)
        traj_y.append(y)

    print("Drawing trajectory...")
    img_trajectories = draw_trajectories(
        traj_x,
        traj_y,
        world_width_m=10.0,
        world_height_m=70.0,
        image_width_px=WIDTH_TRAJECTORY_WINDOW_PX,
        # image_height_px=400
        image_height_px=current_input_image.shape[0]
    )
    # cv2.imwrite("trajectories.png", img_trajectories)

    print("Projecting trajectory onto input image...")
    img_trajectory_projected = draw_trajectory_projected(
        img=current_input_image,
        traj_x=traj_x,
        traj_y=traj_y,
        fx=300.0,
        fy=300.0,
        camera_height_m=1.5,
        camera_pitch_deg=6.0,
    )

    img_output = cv2.hconcat([img_trajectory_projected, img_trajectories])

    print("Saving output image...")
    put_text_with_bg(
        img_output,
        ', '.join(reason_text_list),
        (10, 20),
    )
    # cv2.imwrite(f"trajectory_projected_{frame_id}.png", img_output)
    writer.write(img_output)

writer.release()
cap.release()
