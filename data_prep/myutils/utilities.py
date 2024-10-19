import os
from PIL import Image
import numpy as np
import random
import copy

def unnormbb_vidvrd(bb_data, width, height, round_by=3):
    newbb_data = copy.deepcopy(bb_data)
    print(bb_data, width, height)
    newbb_data['xmin'] = int(round(newbb_data['xmin'],round_by)*width)
    newbb_data['ymin'] = int(round(newbb_data['ymin'],round_by)*height)
    newbb_data['xmax'] = int(round(newbb_data['xmax'],round_by)*width)
    newbb_data['ymax'] = int(round(newbb_data['ymax'],round_by)*height)
    print(newbb_data)
    return newbb_data

class SGSpecialTokens:
    VIDEO_FRAME_ID = "#frameid"
    SG_END = "#sgend"
    SG_SUBJECT = "#subject"
    SG_SUBJECT_ID = "#subid"
    SG_OBJECT = "#object"
    SG_OBJECT_ID = "#objid"
    SG_PREDICATE = "#sgpred"
    SG_BB_START = "#sgbb"
    SG_BB_END = "#sgbbend"
    SG_BB_X1Y1 = "#bbx1y1"
    SG_BB_X2Y2 = "#bbx2y2"
    # SG_START = "#sg"
    # SG_BB_X1 = "#sgx1"
    # SG_BB_X2 = "#sgx2"
    # SG_BB_Y1 = "#sgy1"
    # SG_BB_Y2 = "#sgy2"

    @staticmethod
    def get_tokens():
        members = [attr for attr in dir(SGSpecialTokens) if not callable(getattr(SGSpecialTokens, attr)) and not attr.startswith("__")]
        tokens = []
        for mem in members:
            tokens.append(SGSpecialTokens.__getattribute__(SGSpecialTokens,mem))
        return tokens

def get_block_number_for_frame(frame_idx, frame_blocks):
    frame_block = None
    for block, b_frames in enumerate(frame_blocks):
        if frame_idx in b_frames:
            frame_block = block
            break
    return frame_block

def get_bb_subj_obj(data_root,vid_id,frame_idx,subject_id,object_id):
  sub_bb, obj_bb, mask_size = [], [], None
  try:
    sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=subject_id)
  except FileNotFoundError:
    #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
    pass
  
  try:
    obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=object_id)
  except FileNotFoundError:
    #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
    pass

  return sub_bb, obj_bb, mask_size

def get_frame_range_for_annotations(vid_objects, vid_data):
  min_frame_idx, max_frame_idx = -1, 0
  frames_for_obj = {}
  for vid_obj_idx, vobj in enumerate(vid_objects):
    category = vobj["category"]
    object_id = vobj["object_id"]
    frames_ = getFramesForObject(vid_data=vid_data, Subject_id=object_id)
    if frames_=="None":
        continue
    
    for frame_range in frames_:
      frame_start, frame_end = frame_range

      if f"{category}{object_id}" not in frames_for_obj:
        frames_for_obj[f"{category}{object_id}"] = {
          "frames": []
        }

      frames_for_obj[f"{category}{object_id}"]["frames"].append(frame_range)

      if min_frame_idx ==-1:
          min_frame_idx = frame_start
      if frame_start<=min_frame_idx:
        min_frame_idx = frame_start
      if frame_end>=max_frame_idx:
        max_frame_idx = frame_end

  return min_frame_idx, max_frame_idx, frames_for_obj

def create_batch_frames(vid_data, totalFrames, frame_batch=8):
    ## out of total frames send frames in batch of 8.
    total_frame_batch = int(totalFrames/8)
    remaining_frames = totalFrames%8

    vid_rels = vid_data["relations"]

    frames_to_consider = []
    batch_of_frames = []
    batch_of_frames_rels = []
    total_frame_indices = [i for i in range(totalFrames)]

    rels_for_the_block = []
    batch_rels = []
    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3].copy()
        frame_start, frame_end = frames[0][0], frames[0][1]
        if frame_start>totalFrames:
           continue
        if frame_end>totalFrames:
           continue

        if frame_start not in frames_to_consider:
            frames_to_consider.append(frame_start)
            rels_for_the_block.append(vid_r)

        if len(frames_to_consider)>=8:
            batch_of_frames.append(frames_to_consider)
            batch_rels.append(rels_for_the_block)
            frames_to_consider = []
            rels_for_the_block = []


    if len(rels_for_the_block)>0:
        batch_rels.append(rels_for_the_block)


    # print("frames to consider ", len(frames_to_consider))
    if len(frames_to_consider)>0 and len(frames_to_consider)<8:
        while len(frames_to_consider)<8:

            random_idx = random.choice(total_frame_indices)
            if random_idx not in frames_to_consider:
                frames_to_consider.append(random_idx)

            if len(frames_to_consider)>=8:
                break
        batch_of_frames.append(frames_to_consider)
        
    # total_frame_indices = [i for i in range(totalFrames)]
    # current_frame_batch_idx = 0
    # while current_frame_batch_idx<=total_frame_batch:
    #     start_idx = current_frame_batch_idx * frame_batch
    #     frames_to_infer = total_frame_indices[start_idx:start_idx+frame_batch]
    #     # print(f"T {start_idx}:{start_idx+8} => {frames_to_infer}")
    #     current_frame_batch_idx +=1
    #     if len(frames_to_infer)<frame_batch:
    #         print("less frames batch")
    #         continue
    #     batch_of_frames.append(frames_to_infer)
    # last_batch_remaining = total_frame_indices[-remaining_frames-(frame_batch-remaining_frames):] # add previous batch frames to accomodate n batch
    # batch_of_frames.append(last_batch_remaining)
    # print("batch of frames", batch_of_frames)

    return batch_of_frames, remaining_frames, batch_rels

# def create_batch_frames(totalFrames, frame_batch=8):
#     ## out of total frames send frames in batch of 8.
#     total_frame_batch = int(totalFrames/8)
#     remaining_frames = totalFrames%8

#     batch_of_frames = []

#     total_frame_indices = [i for i in range(totalFrames)]
#     current_frame_batch_idx = 0
#     while current_frame_batch_idx<=total_frame_batch:
#         start_idx = current_frame_batch_idx * frame_batch
#         frames_to_infer = total_frame_indices[start_idx:start_idx+frame_batch]
#         # print(f"T {start_idx}:{start_idx+8} => {frames_to_infer}")
#         current_frame_batch_idx +=1
#         if len(frames_to_infer)<frame_batch:
#             # print("less frames batch")
#             continue
#         batch_of_frames.append(frames_to_infer)

    
#     last_batch_remaining = total_frame_indices[-remaining_frames-(frame_batch-remaining_frames):] # add previous batch frames to accomodate n batch
#     batch_of_frames.append(last_batch_remaining)

#     return batch_of_frames, remaining_frames

def getboundingBoxOftheObject(data_root, vid_id, frame_id, object_id, normlize_bb=True, dataset="vidor"):
    mask_name = os.path.join(data_root, dataset, 'masks', vid_id, f'{str(frame_id).zfill(4)}.png')
    mask = Image.open(mask_name)
    mask = np.array(mask)

    segmentation = np.where(mask == object_id)
    mask_h, mask_w = mask.shape[0],mask.shape[1]
    # maskbb = np.zeros(shape=(mask_h,mask_w,3), dtype=np.uint8)

    # Bounding Box
    bbox = []
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        if normlize_bb:
           x_min = round(x_min/mask_w,3)
           x_max = round(x_max/mask_w,3)
           y_min = round(y_min/mask_h,3)
           y_max = round(y_max/mask_h,3)

        bbox = [x_min, y_min, x_max, y_max]
        # print(bbox)
        # cv2.rectangle(maskbb, (x_min, y_min), (x_max, y_max), (36,255,12), 2)

    return bbox,[mask_h, mask_w]

def getFramesForObject(vid_data, Subject_id):
    vid_rels = vid_data["relations"]
    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        # rel = vid_r[2]
        frames_ = vid_r[3].copy()
        if Subject_id==sub or Subject_id==obj:
            return frames_
    return "None"

def unnormbb(pred_box, mask):
    pred_box[0] = int(pred_box[0]*mask.shape[1])
    pred_box[2] = int(pred_box[2]*mask.shape[1])
    pred_box[1] = int(pred_box[1]*mask.shape[0])
    pred_box[3] = int(pred_box[3]*mask.shape[0])
    return pred_box


def parse_bb_from_string(str_data):
    # "[0.048, 0.0, 0.517, 0.997]"
    bb = []
    str_data = str_data.strip("</s>").strip("[").strip("]").split(",")
    if len(str_data)<4:
        return []
    for bb_coord in str_data:
        try:
            bb.append(round(float(bb_coord),3))
        except ValueError:
            return []
    return bb

def parse_sg_data(pred_sg_str_data):
    pred_sgs = []
    predictions = pred_sg_str_data.strip("</s>").split(";")

    for pred in predictions:
        if len(pred)<30:
            continue

        # subPredObj, pred_Frame = pred.split("_")
        # pred_Frame = pred_Frame.strip("[").strip("]")
        pred_Frame = 0

        triplates = pred.split(":")
        if len(triplates)==3:
            subj, predi, obj = triplates

            subj_data = subj.split("-")
            if len(subj_data)<3:
                continue
            subj_id = f"{subj_data[0].strip('[').strip(']')}-{subj_data[1]}"
            subj_bb = parse_bb_from_string(subj_data[2])

            obj_data = obj.split("-")
            
            if len(obj_data)<3:
                continue

            obj_id = f"{obj_data[0]}{obj_data[1]}"
            
            obj_bb = parse_bb_from_string(obj_data[2])

            sg = {
                "subject": {
                    "id": subj_id,
                    "bbox": subj_bb
                },
                "predicate": predi,
                "object":{
                    "id": obj_id,
                    "bbox": obj_bb
                },
                "uni_frame_idx": pred_Frame
            }

            pred_sgs.append(sg)

    return pred_sgs 


def validate_model_response(model_response):
    validation_flags = []

    if "{" not in model_response:
        validation_flags.append(1)

    if "}" not in model_response:
        validation_flags.append(1)
    
    for i in range(8):
        if not f"Frame {i}" in model_response:
            validation_flags.append(1)

    # print("validation flags ", validation_flags)
    if sum(validation_flags)>0:
        return False
    
    return True

def clean_prediction_data(model_response, val_id, block_id):
    prediction_data = model_response[val_id][f"{block_id}"].copy()
    prediction_data = prediction_data["triplets"].strip("</s>")

    # print("#"*5)
    # print(prediction_data)
    # print("#"*5)

    if validate_model_response(model_response=prediction_data):

        # print("First token ###==>", prediction_data[0])
        # if prediction_data[0].strip()!="{":
        #     if "Frame 0" in prediction_data:
        token_cnt = prediction_data.count("Frame 0")  # if more than 1 reponse repeated
        # print("token count ", token_cnt)
        if token_cnt>1:
            token_idx = prediction_data.index("}")
            prediction_data = prediction_data[0:token_idx+1]
            # print("NEW STRING########> ", prediction_data)

        model_res_len = len(prediction_data)
        end_idx = prediction_data.index("}")
        if model_res_len!=end_idx:
            prediction_data = prediction_data[0:end_idx+1]
        
        if prediction_data[-1]!="}":
            prediction_data_split = prediction_data.split(";")
            last_element = prediction_data_split[-1]
            last_element_spilit = last_element.split(":")
            if len(last_element_spilit)<3:
                del prediction_data_split[-1]
            prediction_data = "".join(prediction_data_split)
            prediction_data += "'}"

        for i in range(8):
            prediction_data = prediction_data.replace(f"Frame {i}", f"{i}")
        
        FrameLevelPredictions = eval(prediction_data)

        return FrameLevelPredictions
    
    return None

def remove_duplicates(frame_level_prediction_for_block):

    # print("#######RM DP#######", frame_level_prediction_for_block)
    all_over_triplates = []
    for frame, data in frame_level_prediction_for_block.items():
        data_split = data.split("[")
        triplates = []
        for i_, data_ in enumerate(data_split):
            data_ = data_.strip("]").strip(";")

            sub_pred_obj = data_.split(":")
            if len(sub_pred_obj)!=3:
                continue
            if sub_pred_obj not in triplates:
                for spoidx, subpreobj in enumerate(sub_pred_obj):
                    sub_pred_obj[spoidx] = subpreobj.strip("]").strip(";")
                triplates.append(sub_pred_obj)

            if sub_pred_obj not in all_over_triplates:
                all_over_triplates.append(sub_pred_obj)
        
        frame_level_prediction_for_block[frame] = triplates
    return frame_level_prediction_for_block, all_over_triplates


prompts_list = {
    
    "summary": ["Describe the video in detail",
                "What is happening in the video?",
                "What is the central narrative or story in the video?",
                "What is the purpose or goal of the video?",
                "What are the key takeaways or lessons from the video?"
                ],

    "identify_subject_objects": [
                        "List the objects present in the video",
                        "What objects, items, or elements appear prominently?", 
                        "Identify any significant objects in the video.",
                        "What objects are visible in the video?",
                        "List the main objects featured in the video.",
                        "what are the main objects featured in the video?"
                        ],
    "identify_predicates": [
                            "List the actions, movements or placements of the objects in the scene.",
                            "Describe any interactions between people or objects in the video.",
                            "Describe any significant gestures or interactions between objects in the scene",
                            "How subjects and objects relates to each other in the video?",
                            "How do the objects interact with their environment in the video?",
                            "Describe any notable physical interactions between objects in the video.",
                            "Describe any interactions that highlight the relationships between objects.",
                            "What actions or events take place in the video?",
                          ],
    "SGG": [
       "Generate frame-by-frame scene graph for the provided video",
       "Provide frame-by-frame Scene graph triplets in the form of [Subject-id:Object-id:Predicate]",
       "Generate scene graph for the provided video",
       "Provide scene graph for the provided video",
       "Identify subjects, predicates and objects frame-by-frame in the provided video"
    ],

    "SGG_image": [
       "Generate scene graph for the provided image",
       "Provide Scene graph triplets in the form of [Subject-id:Predicate:Object-id] for the provided image",
       "Generate scene graph for the provided image",
       "Provide scene graph for the provided image",
       "Identify subjects, predicates and objects in the provided image"
    ],

    "SGG_with_bb": [
       "Generate frame-by-frame scene graph for the provided video along with bounding box of each objects",
       "Provide frame-by-frame Scene graph triplets in the form of [Subject-id-[min_x,min_y,max_x,max_y]:Predicate:Object-id-[min_x,min_y,max_x,max_y]]",
       "Generate scene graph for the provided video along with bounding box of each objects",
       "Provide scene graph for the provided video with bounding box location of each objects",
       "Identify Subjects, Predicates and Objects frame-by-frame in the provided video, also provide bounding box location of each subject and object"
    ],

    "sg_localization": [
      "Provide bounding box location of [{sub}:{rel}:{obj}] in frame {frame_idx} of the provided video" # {} to be replaced by actual value
      #"Provide bounding box location of [{sub}:{rel}:{obj}]" # {} to be replaced by actual value
    ],

    "sg_localization_image": [
      "Provide bounding box location of [{sub}:{rel}:{obj}]" # {} to be replaced by actual value
    ],

    "predict_predicate": [
      "What is the relationship between [{sub}:{obj}] in the video. Use only the provided lists for predicates. Predicates: {predicates}" # {} to be replaced by actual value
    ],

    # """
    # Generate frame-by-frame scene graph for the provided video
    #    Use the following list to select the object {}, 
    #    the following list to select the subject {}, 
    #    and the follwing list to select the predicate {}.
    # """

    # "triplet_prompt": [
    #   """You are given a list of predefined subjects, objects, and predicates. Your task is to predict scene graph triplets in the format [Subject:Object:Predicate] based on the given scene description in the video. Use only the provided lists for objects, and predicates.
    #   Subjects: {subjects}
    #   Objects: {objects}
    #   Predicates: {predicates}
    #   """
    # ]
    "triplet_prompt": [
      """You are given a list of predefined subjects, objects, and predicates. Your task is to predict scene graph triplets in the format [Subject-id:Predicate:Object-id] based on the given scene in the video. Use only the provided lists for subject, objects, and predicates. \n\
      Subjects: {subjects} \n\
      Predicates: {predicates} \n\
      Objects: {objects} \n\
      """
    ]


}


def getConvBlock(value,conv_type="human", media_type="<image>", add_media_token=False):
   assert conv_type=="human" or conv_type=="gpt"
   assert media_type=="<image>" or media_type=="<video>"
   conv = {"from": conv_type, "value": f"{value}"}
   if add_media_token:
      conv["value"] = f"{media_type}\n{value}"
   else:
      conv["value"] = f"{value}" 

   return conv

def getPromptTemplate(media_path, media_type="image"):
  assert media_type=="image" or media_type=="video"
  Prompt = {
          "id": "TobeUpdated",
          f"{media_type}": f"{media_path}",
          "conversations": [],
          "frame_indices": [],  # selected indices will be passed to model for train and test
          "total_frames": "",
  }
  return Prompt


def getRandomPrompt(key="summary", static=False):
    if static:
       return prompts_list[key][0]
    return random.choice(prompts_list[key])

def getFramesForObject(vid_data, Subject_id):
    vid_rels = vid_data["relations"]
    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        # rel = vid_r[2]
        frames_ = vid_r[3].copy()
        if Subject_id==sub or Subject_id==obj:
            return frames_
    return "None"


def getbbcenter(bb):
   if len(bb)<4:
    return []
   x1,y1,x2,y2 = bb
   bb_w = (x2 - x1)/2
   bb_h = (y2 - y1)/2
   xcenter = x1 + bb_w
   ycenter = y1 + bb_h
   return [round(xcenter,3), round(ycenter,3)]

def getListofCategoryString(data_root, vid_objects, vid_data, addObjectId=False, addFrames=False, addBB=False , uniform_sampling_idx=8):
    
    AnswerString = ""
    frame_indices = []
    total_frames = vid_data["meta"]["num_frames"]
    """V11 implementation
    [X] Select frames which covers all objects, avoid repetations
    """

    frames_where_obj_is_present = {}
    min_frame_idx, max_frame_idx, frames_for_obj = get_frame_range_for_annotations(vid_objects, vid_data)

    for frame_idx  in range(min_frame_idx, max_frame_idx+1):
      if frame_idx>total_frames:
         continue

      if frame_idx not in frames_where_obj_is_present.keys():
        frames_where_obj_is_present[frame_idx] ={
          "objects_present": [],
          "object_bb": [],
          "object_cnt": 0
        }

      for vid_obj_idx, vobj in enumerate(vid_objects):
        category = vobj["category"]
        object_id = vobj["object_id"]
        
        try:
          sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frame_idx, object_id=object_id)
        except FileNotFoundError:
          pass

        if sum(sub_bb)>0:
          frames_where_obj_is_present[frame_idx]["objects_present"].append(vobj)
          frames_where_obj_is_present[frame_idx]["object_bb"].append(sub_bb)
          frames_where_obj_is_present[frame_idx]["object_cnt"] +=1

    # Take frames with more objects count first
    frames_with_obj_cnt = [(frames_where_obj_is_present[f_idx]["object_cnt"], f_idx) for f_idx in frames_where_obj_is_present]
    frames_with_obj_cnt = sorted(frames_with_obj_cnt,reverse=True)

    objects_added = []

    """
    Frame wise
    AnswerString = {
      0: "floor-1, wall-1, pillow-4",
      1: "floor-1, wall-1, shelf-4"
      .
      .
      7: "obj1,obj2"
    }
    """

    AnswerString += "{"

    for f_obj_idx, f_obj_cnt in enumerate(frames_with_obj_cnt):
      cnt_,f_idx = f_obj_cnt
      data = frames_where_obj_is_present[f_idx]

      AnswerString += f"{f_obj_idx}:"
      AnswerString +="'"  # start the list of objects string by "'"

      objects_present = data["objects_present"]
      objects_bb = data["object_bb"]

      frame_indices.append(f_idx) # use frame indices where object annotations are present

      for oidx, obj in enumerate(objects_present):
        category = obj["category"]
        object_id = obj["object_id"]

        # object_name_id = f"{category}-{object_id}"
        # if object_name_id not in objects_added:
        #   """This ensures unique objects in the list"""
        #   objects_added.append(object_name_id)

        AnswerString += f"{category}"

        if addObjectId:
          AnswerString += f"-{object_id}"

        if addBB:
          AnswerString += f"-{objects_bb[oidx]}"
        if addFrames:
          AnswerString += f"_[{f_idx}]"

        if oidx!=len(objects_present)-1:
          AnswerString +=","
        else:
          AnswerString +="'"  # finish the list of objects string by "'"
        
        if f_obj_idx>6:
           # TODO: some objects which appears in low count, will not be taken due to object density
           # In order to resolve this issue, need to accomodate all frames in 8 frames
           break
        
        if f_obj_idx!=len(frames_with_obj_cnt)-1:
           AnswerString += f"," # end of current key in dict


    AnswerString += "}"

    return AnswerString, frame_indices


def getboundingBoxOftheObject(data_root, vid_id, frame_id, object_id, normlize_bb=True, dataset="vidor"):
    mask_name = os.path.join(data_root, dataset, 'masks', vid_id, f'{str(frame_id).zfill(4)}.png')
    mask = Image.open(mask_name)
    mask = np.array(mask)

    segmentation = np.where(mask == object_id)
    mask_h, mask_w = mask.shape[0],mask.shape[1]
    # maskbb = np.zeros(shape=(mask_h,mask_w,3), dtype=np.uint8)

    # Bounding Box
    bbox = []
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        if normlize_bb:
           x_min = round(x_min/mask_w,3)
           x_max = round(x_max/mask_w,3)
           y_min = round(y_min/mask_h,3)
           y_max = round(y_max/mask_h,3)

        bbox = [x_min, y_min, x_max, y_max]
        # print(bbox)
        # cv2.rectangle(maskbb, (x_min, y_min), (x_max, y_max), (36,255,12), 2)

    return bbox,[mask_h, mask_w]



def get_frame_range_for_annotations(vid_objects, vid_data,):
  min_frame_idx, max_frame_idx = -1, 0
  frames_for_obj = {}
  for vid_obj_idx, vobj in enumerate(vid_objects):
    category = vobj["category"]
    object_id = vobj["object_id"]
    frames_ = getFramesForObject(vid_data=vid_data, Subject_id=object_id)
    if frames_=="None":
        continue
    
    for frame_range in frames_:
      frame_start, frame_end = frame_range

      if f"{category}{object_id}" not in frames_for_obj:
        frames_for_obj[f"{category}{object_id}"] = {
          "frames": []
        }

      frames_for_obj[f"{category}{object_id}"]["frames"].append(frame_range)

      if min_frame_idx ==-1:
        min_frame_idx = frame_start
      if frame_start<=min_frame_idx:
        min_frame_idx = frame_start
      if frame_end>=max_frame_idx:
        max_frame_idx = frame_end

  return min_frame_idx, max_frame_idx, frames_for_obj