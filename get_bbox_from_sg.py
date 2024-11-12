import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import supervision as sv
from typing import List, Tuple
from tqdm import tqdm
import cv2
import numpy as np
from copy import deepcopy
import argparse

def annotate_bbox(image_source: np.ndarray, boxes: torch.Tensor, scores: torch.Tensor, labels: List[str]) -> np.ndarray:

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    if not isinstance(labels, list):
        labels = [labels]
    
    if not isinstance(scores, list):
        scores = [scores]

    detections = sv.Detections(xyxy=boxes)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(labels, scores)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = image_source.copy()
    # annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

import numpy as np
import cv2
import torch
from typing import List, Tuple

def annotate_with_scene_graph(
    image_source: np.ndarray,
    scene_graph: List[Tuple[List, str, List]],
) -> np.ndarray:
    
    annotated_frame = image_source.copy()
    text_positions = {}  # Dictionary to store text positions to avoid overlap

    # Loop through each entry in the scene graph and annotate nodes and edges
    for (subject, rel, obj) in scene_graph:
        sub_label, _, sub_bbox, sub_score = subject
        obj_label, _, obj_bbox, obj_score = obj

        # Draw subject and object bounding boxes and labels
        cv2.rectangle(
            annotated_frame,
            (int(sub_bbox[0]), int(sub_bbox[1])),
            (int(sub_bbox[2]), int(sub_bbox[3])),
            (0, 255, 0), 2
        )
        cv2.putText(
            annotated_frame,
            f"{sub_label} {sub_score:.2f}",
            (int(sub_bbox[0]), int(sub_bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        cv2.rectangle(
            annotated_frame,
            (int(obj_bbox[0]), int(obj_bbox[1])),
            (int(obj_bbox[2]), int(obj_bbox[3])),
            (255, 0, 0), 2
        )
        cv2.putText(
            annotated_frame,
            f"{obj_label} {obj_score:.2f}",
            (int(obj_bbox[0]), int(obj_bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        # Draw relation line between subject and object
        start_point = (
            int((sub_bbox[0] + sub_bbox[2]) / 2),
            int((sub_bbox[1] + sub_bbox[3]) / 2)
        )
        end_point = (
            int((obj_bbox[0] + obj_bbox[2]) / 2),
            int((obj_bbox[1] + obj_bbox[3]) / 2)
        )
        cv2.line(annotated_frame, start_point, end_point, (0, 255, 255), 2)

        # Calculate the midpoint for the relation label
        mid_point = (
            int((start_point[0] + end_point[0]) / 2),
            int((start_point[1] + end_point[1]) / 2)
        )

        # Check for overlap and adjust position if necessary
        offset_y = 0
        while (mid_point[0], mid_point[1] + offset_y) in text_positions:
            offset_y += 15  # Adjust downward if overlap is detected

        # Draw the relation label at the adjusted position
        adjusted_position = (mid_point[0], mid_point[1] + offset_y)
        cv2.putText(
            annotated_frame,
            rel,
            adjusted_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

        # Store the adjusted position in text_positions to track placed texts
        text_positions[(mid_point[0], mid_point[1] + offset_y)] = True

    return annotated_frame

class SceneGraphDetector:
    def __init__(self, model_path="IDEA-Research/grounding-dino-base", device="cuda"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device)
        self.device = device

    def predict_bbox(self, images, texts, box_threshold=0.2, text_threshold=0.2, visualize=False):
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, text=texts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1] for image in images]
        )

        if visualize:
            annotated_images = []
            for i, image in enumerate(images):
                annotated_images.append(annotate_bbox(np.array(image), results[i]["boxes"], results[i]["scores"], results[i]["labels"]))
            return results, annotated_images
        else:
            return results
        
    @torch.no_grad()
    def predict_bbox_for_scene_graph_ag(self, scene_graph_dict, data_dir="/groups/sernam/datasets/ActionGenome/ActionGenome/frames", box_threshold=0.2, text_threshold=0.2, batch_size=16):

        # TODO: Right now we predict bbox for every subject and object in every triplet for a given frame. 
        # But in prior works we only calculate them once since they are unique. 
        # We can either do the same (e.g. use spatial to get obj bbox and other two for person),
        # or we can do a majority vote 

        flattened_scene_graph = []
        
        for video_name, item in scene_graph_dict.items():
            for block_id, block in item.items():
                frames = block["frames"]
                num_gt_frames = len(frames)
                num_pred_frames = len(block["triplets"])
                num_frames = num_gt_frames
                if num_gt_frames != num_pred_frames:
                    print(f"Number of gt frames({num_gt_frames}) does not match the number of predicted frames ({num_pred_frames}). Setting the number of frames to the minimum.")
                    num_frames = min(num_gt_frames, num_pred_frames)
                    
                for frame_num, frame_triplets in zip(frames[:num_frames], block["triplets"][:num_frames]):

                    image_name = f"{video_name}/{str(frame_num).zfill(6)}.png"

                    for sub, rel, obj in frame_triplets:
                        if rel == 'other relationship':
                            prompt_sub = f"a {sub}."
                            prompt_obj = f"a {obj}."
                        else:
                            prompt_sub = f"the {sub} {rel} {obj}."
                            prompt_obj = f"the {obj} being {rel} {sub}."

                        flattened_scene_graph.append({"prompt": prompt_sub,
                                                        "label": sub,
                                                        "rel": rel,
                                                        "image_name": image_name,
                                                        "video_name": video_name,
                                                        "block_id": block_id,
                                                        "num_frames": num_gt_frames})
                        flattened_scene_graph.append({"prompt": prompt_obj,
                                                        "label": obj,
                                                        "rel": rel,
                                                        "image_name": image_name,
                                                        "video_name": video_name,
                                                        "block_id": block_id,
                                                        "num_frames": num_gt_frames})


        results = {}

        frame_triplets = []
        # batch inference
        for i in tqdm(range(0, len(flattened_scene_graph), batch_size)):

            images = [Image.open(os.path.join(data_dir, sample['image_name'])) for sample in flattened_scene_graph[i:i+batch_size]]
            result_batch = self.predict_bbox(images=images, texts=[sample['prompt'] for sample in flattened_scene_graph[i:i+batch_size]], box_threshold=box_threshold, text_threshold=text_threshold)

            sub = None

            for batch_ind, res in enumerate(result_batch):
                current_idx = i + batch_ind
                
                video_id = flattened_scene_graph[current_idx]['video_name']
                block_id = flattened_scene_graph[current_idx]['block_id']
                if video_id not in results:
                    results[video_id] = {}
                if block_id not in results[video_id]:
                    results[video_id][block_id] = {"frames": scene_graph_dict[video_id][block_id]["frames"], "triplets": []}

                scores, pred_labels, bboxes = res['scores'].cpu(), res['labels'], res['boxes']

                highest_score = torch.tensor(-1)
                highest_score_bbox = torch.tensor([0,0,0,0])
                highest_label = None

                for j in range(len(scores)):
                    # grounding-dino's label could be 'a person looking at towel' for the towel object. 
                    # In case we want person, we just want to look at the first two words
                        # some objects could be "sofa/couch". if either of them is in the label, we consider it as a match
                    if any([label for label in flattened_scene_graph[current_idx]['label'].split("/") if label in pred_labels[j].split(" ")[:3]]) and scores[j] > highest_score:
                        highest_score = scores[j]
                        highest_score_bbox = bboxes[j]
                        highest_label = pred_labels[j]
                if sub is None:
                    sub = (flattened_scene_graph[current_idx]['label'], highest_label, [round(x, 3) for x in highest_score_bbox.cpu().tolist()], round(highest_score.item(), 2))
                else:
                    triplet = (sub, flattened_scene_graph[current_idx]['rel'], (flattened_scene_graph[current_idx]['label'], highest_label, [round(x, 3) for x in highest_score_bbox.cpu().tolist()], round(highest_score.item(), 2)))
                    if len(frame_triplets) < flattened_scene_graph[current_idx]['num_frames']:
                        frame_triplets.append(triplet)
                    else:
                        results[video_id][block_id]["triplets"].append(frame_triplets)
                        frame_triplets = [triplet]
                    sub = None


                # TODO: I'm not sure if it's legit to do that... feels a little cheating
                # # in some frames the object may not be detected, use the adjacent frame's bbox
                # for frame_i in range(len(block_triplets_with_bbox)):
                #     for triplet_i in range(len(block_triplets_with_bbox[frame_i])):
                #         # first check if there are same sub and obj pair in the frame
                #         if block_triplets_with_bbox[frame_i][triplet_i][0][2] is None:
                #             find_bbox = False
                #             # look back
                #             for k in range(frame_i-1, -1, -1):
                #                 for triplets in block_triplets_with_bbox[k]:
                #                     if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                     and triplets[2][0] == block_triplets_with_bbox[frame_i][triplet_i][2][0] \
                #                     and triplets[0][2] is not None:
                #                         block_triplets_with_bbox[frame_i][triplet_i][0][2] = triplets[0][2]
                #                         find_bbox = True
                #                         break
                #                 if find_bbox:
                #                     break
                #             if find_bbox:
                #                 continue
                #             # look forward
                #             for k in range(frame_i+1, len(block_triplets_with_bbox)):
                #                 for triplets in block_triplets_with_bbox[k]:
                #                     if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                     and triplets[2][0] == block_triplets_with_bbox[frame_i][triplet_i][2][0] \
                #                     and triplets[0][2] is not None:
                #                         block_triplets_with_bbox[frame_i][triplet_i][0][2] = triplets[0][2]
                #                         find_bbox = True
                #                         break   
                #                 if find_bbox:
                #                     break
                #             if find_bbox:
                #                 continue
                #             # check if subject is still not detected
                #             if block_triplets_with_bbox[frame_i][triplet_i][0][2] is None:
                #                 for k in range(frame_i-1, -1, -1):
                #                     for triplets in block_triplets_with_bbox[k]:
                #                         if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                         and triplets[0][2] is not None:
                #                             block_triplets_with_bbox[frame_i][triplet_i][0][2] = triplets[0][2]
                #                             find_bbox = True
                #                             break
                #                     if find_bbox:
                #                         break
                #                 if not find_bbox:
                #                     for k in range(frame_i+1, len(block_triplets_with_bbox)):
                #                         for triplets in block_triplets_with_bbox[k]:
                #                             if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                             and triplets[0][2] is not None:
                #                                 block_triplets_with_bbox[frame_i][triplet_i][0][2] = triplets[0][2]
                #                                 find_bbox = True
                #                                 break
                #                         if find_bbox:
                #                             break
                # for frame_i in range(len(block_triplets_with_bbox)):
                #     for triplet_i in range(len(block_triplets_with_bbox[frame_i])):
                #         # first check if there are same sub and obj pair in the frame
                #         if block_triplets_with_bbox[frame_i][triplet_i][2][2] is None:
                #             find_bbox = False
                #             # look back
                #             for k in range(frame_i-1, -1, -1):
                #                 for triplets in block_triplets_with_bbox[k]:
                #                     if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                     and triplets[2][0] == block_triplets_with_bbox[frame_i][triplet_i][2][0] \
                #                     and triplets[0][2] is not None:
                #                         block_triplets_with_bbox[frame_i][triplet_i][2][2] = triplets[0][2]
                #                         find_bbox = True
                #                         break
                #                 if find_bbox:
                #                     break
                #             if find_bbox:
                #                 continue
                #             # look forward
                #             for k in range(frame_i+1, len(block_triplets_with_bbox)):
                #                 for triplets in block_triplets_with_bbox[k]:
                #                     if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                     and triplets[2][0] == block_triplets_with_bbox[frame_i][triplet_i][2][0] \
                #                     and triplets[0][2] is not None:
                #                         block_triplets_with_bbox[frame_i][triplet_i][2][2] = triplets[0][2]
                #                         find_bbox = True
                #                         break   
                #                 if find_bbox:
                #                     break
                #             if find_bbox:
                #                 continue
                #             # check if subject is still not detected
                #             if block_triplets_with_bbox[frame_i][triplet_i][2][2] is None:
                #                 for k in range(frame_i-1, -1, -1):
                #                     for triplets in block_triplets_with_bbox[k]:
                #                         if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                         and triplets[0][2] is not None:
                #                             block_triplets_with_bbox[frame_i][triplet_i][2][2] = triplets[0][2]
                #                             find_bbox = True
                #                             break
                #                     if find_bbox:
                #                         break
                #                 if not find_bbox:
                #                     for k in range(frame_i+1, len(block_triplets_with_bbox)):
                #                         for triplets in block_triplets_with_bbox[k]:
                #                             if triplets[0][0] == block_triplets_with_bbox[frame_i][triplet_i][0][0] \
                #                             and triplets[0][2] is not None:
                #                                 block_triplets_with_bbox[frame_i][triplet_i][2][2] = triplets[0][2]
                #                                 find_bbox = True
                #                                 break
                #                         if find_bbox:
                #                             break

        return results
    
    def visualize_grounded_sg(self, scene_graph_dict, video_name, data_dir="/groups/sernam/datasets/ActionGenome/ActionGenome/frames"):
        """
            param: scene_graph_dict: in the form of 
            [
                [
                    [[sub1, bbox, score], rel, [obj1, bbox, score]],
                    [[sub2, bbox, score], rel, [obj2, bbox, score]] ...
                ],
                ...
            ]
 
        """
        videos = []
        for block_id in scene_graph_dict[video_name]:
            scene_graph = scene_graph_dict[video_name][str(block_id)]
            image_paths = [f"{video_name}/{str(frame_num).zfill(6)}.png" for frame_num in scene_graph['frames']]

            annotated_images = []
            scene_graph_images = []
            # print the scene graph text also onto a white image
            for frame_sg, frame_path in zip(scene_graph['triplets'], image_paths):
                image = Image.open(os.path.join(data_dir, frame_path))
                annotated_images.append(annotate_with_scene_graph(np.array(image), frame_sg))
                white_image = np.ones_like(np.array(image)) * 255
                for idx, triplets in enumerate(frame_sg):
                    cv2.putText(
                        white_image,
                        f"{triplets[0][0]} {triplets[1]} {triplets[2][0]}",
                        (10, 25 * (idx+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
                    )
                scene_graph_images.append(white_image)
                
            # concat all images horizontally and return
            annotated_video = np.concatenate(annotated_images, axis=1)
            scene_graph = np.concatenate(scene_graph_images, axis=1)
            videos.append(np.concatenate([annotated_video, scene_graph], axis=0))
        return videos

def load_multiple_jsons_from_folder(folder_path):
    all_jsons = [json.load(open(os.path.join(folder_path, f), "r")) for f in os.listdir(folder_path) if f.endswith("inference_val.json")]
    combined = {}
    for j in all_jsons:
        combined.update(j)
    return combined

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get bbox from scene graph')
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="/groups/sernam/datasets/ActionGenome/ActionGenome/frames")
    args = parser.parse_args()

    pred_sg = load_multiple_jsons_from_folder(args.result_path)
    if os.path.isdir(args.result_path):
        folder_path = args.result_path
    else:
        folder_path = os.path.dirname(args.result_path)
    
    sg_detector = SceneGraphDetector()
    split = 0
    total = 5

    # import random
    # # random_subset = random.sample(pred_sg.keys(), 5)
    # random_subset = ['4G3ZF.mp4', '136V6.mp4', 'U5QJR.mp4', 'BQ963.mp4']
    # print(random_subset)
    # subset = {k: v for k, v in pred_sg.items() if  k in random_subset}

    subset = [k for i, k in enumerate(pred_sg.keys()) if i % total == split]
    subset = {k: v for k, v in pred_sg.items() if  k in subset}
    new_pred_sg = sg_detector.predict_bbox_for_scene_graph_ag(subset, data_dir=args.data_dir)

    json.dump(new_pred_sg, open(os.path.join(folder_path, f"ActionGnome_inference_val_bbox_{split}-{total}.json"), "w"), indent=4)

    for video in subset:
        visualized = sg_detector.visualize_grounded_sg(new_pred_sg, video, data_dir=args.data_dir)
        # save the visualized image
        for i, v in enumerate(visualized):
            cv2.imwrite(f"outputs/{video}_{i}.png", v)