from prompt_magic import TaskDescription, ActualQuestion, InContextExamples, ListOfObjectsAndPredicates as objectspredicates
# import importlib
# importlib.reload(TaskDescription)
# importlib.reload(ActualQuestion)
# importlib.reload(InContextExamples)
# importlib.reload(objectspredicates)

v14_Task_description_AG_triplets_ZS = TaskDescription.Task_description_v14_AG_with_list_GPT


v14_Task_description_AG_triplets_sgcls = TaskDescription.Task_description_v14_ZS_AG_sgcls_short
v14_Task_description_AG_triplets_precls  = TaskDescription.Task_description_v14_ZS_AG_predcls

v14_Task_description_vidor_quadruplets_without_obj_list = TaskDescription.Task_description_v14_vidor_quadruplets_with_list_GPT
v14_prompt_vidor_quadruplets_withIds_without_obj_list = f"""{v14_Task_description_vidor_quadruplets_without_obj_list}"""

v14_Task_description_vidor_triplets_without_obj_list = TaskDescription.Task_description_v14_vidor_triplets_with_list_GPT
v14_prompt_vidor_triplets_withIds_without_obj_list = f"""{v14_Task_description_vidor_triplets_without_obj_list}"""

v14_Task_description_vidvrd_without_list = TaskDescription.Task_description_v14_vidvrd_with_list_GPT
v14_prompt_vidvrd_withIds_without_list = f"""{v14_Task_description_vidvrd_without_list}"""

v14_Task_description_vidvrd_with_list = TaskDescription.Task_description_v14_vidvrd_with_list_GPT
v14_prompt_vidvrd_withIds_with_list = f"""{v14_Task_description_vidvrd_with_list}"""

V13_TaskDescription_withids_temporal = TaskDescription.Task_description_v13_with_ids_temporal
v13_prompt_withIds_temporal = f"""{V13_TaskDescription_withids_temporal}"""

V13_TaskDescription_vidvrd_withids = TaskDescription.Task_description_v13_vidvrd_sam_with_list
v13_prompt_vidvrd_withIds = f"""{V13_TaskDescription_vidvrd_withids}"""

V13_TaskDescription_withids = TaskDescription.Task_description_v13_sam_with_list
v13_prompt_withIds = f"""{V13_TaskDescription_withids}"""

V13_TaskDescription_sam = TaskDescription.Task_description_v13_sam
v13_prompt_sam = f"""{V13_TaskDescription_sam}"""

V13_TaskDescription_ = TaskDescription.Task_description_v13
v13_prompt = f"""{V13_TaskDescription_}"""

V12_TaskDescription_ = TaskDescription.Task_description_v12
v12_prompt = f"""{V12_TaskDescription_}"""

V11_TaskDescription_ = TaskDescription.Task_description_v11
v11_prompt = f"""{V11_TaskDescription_}"""

V10_TaskDescription_ = TaskDescription.Task_description_v10
v10_prompt = f"""{V10_TaskDescription_}"""

V9_TaskDescription_ = TaskDescription.Task_description_v9
v9_prompt = f"""{V9_TaskDescription_}
Note: There are 8 frames in the video.
"""

V8_TaskDescription_ = TaskDescription.Task_description_v8
v8_prompt = f"""{V8_TaskDescription_}
Note: There are 8 frames in the video.
"""

V7_TaskDescription_ = TaskDescription.Task_description_v7
v7_prompt = f"""{V7_TaskDescription_}
Note: There are 8 frames in the video.
"""

v6_prompt = f"""{TaskDescription.Task_description_v6}
Note: There are 8 frames in the video.
"""

v5_prompt  =f"""{TaskDescription.Task_description_v5}
Note: There are 8 frames in the video.
"""

v3_prompt = f"""
    Your task is to generate scene graph for the provided video, to do so follow the steps below.
    1. Describe the given video in detail manner (e.g focus on the background, foreground, main activities and all the objects present in the video), while doing so capture actors, their actions, their apperance and how they are intracting and placed with other objects in the environment.
    2. From the obtained description in step-1 your task is to 
        2.1 Extract subject(actors) and objects(affected by actors) from step-1 who are performing or receiving some actions and describe their spatial location with respect to other objects and assign them a random #id to disntinguish each and every objects in the description. (e.g. deer-1, table-9, grass-0, floor-9 etc.)
        2.2 Capture predicates which are associated with the subject(actors) which can be verbs or spatial location predicates (e.g. Sitting, Standing, Holding, Stand next to, In front, Stand behind etc.)
        2.3 Finally put everything togather and provide list of triplets in the format [subject-#id, predicate, object-#id].
    3. Align open vocabulary triples obtained in previous step to close vocabulary triplets by using provided list of subjects/objects and predicates
        - For example, if triplet from step 2.3 is [person-8, standing next, woman-4] and given list of subject/objects are [adult, grass, mountains, cow] and predicates are [besides, on top, below, holding], then update the triplet as [adult-8,besides,adult-4]

    Example 1:
    Descrption of the video: A which cat chasing the black mouse on the floor, while the cat is behind the mouse, a yellow dog sitting on the corner starts chasing cat.
    2.1 subject(actors): [cat-3, mouse-2, dog-0, corner-1, dog-0, floor-7]
    2.2 predicates: [chasing, sitting on, behind]
    2.3 triplets: [cat-3, on, floor-7];[cat-3,chasing, mouse-2];[dog-0,sitting on,corner-1];[mouse-2,on,floor-7];[dog-0,on,floor-7];[dog-0,chasing,cat-3];[cat-3,behind,mouse-2];[dog-0:behind:cat-3]

    3. Provided list of objects [rat, house_cat] and predicates [run behind, move front], updated triplets #sg_start [cat-3, on, floor-7];[cat-3,chasing,rat-2];[dog-0,sitting on,corner-1];[rat-2,on,floor-7];[dog-0,on,floor-7];[dog-0,chasing,cat-3];[cat-3,run behind,rat-2];[dog-0:run behind:cat-3] #sg_end
    
    Now follow above steps to provided scene graph triples, use below list of objects and predicates for Step-3

    List of objects: {objectspredicates.ListOfObjectsAndPredicates['birthdayboy']['objects']}
    List of predicates: {objectspredicates.ListOfObjectsAndPredicates['birthdayboy']['predicates']}
    """


V2_TaskDescription_ = TaskDescription.Task_description_v5
# V2_TaskDescription_ = V2_TaskDescription_.replace("{List_of_objects}", objectspredicates.ListOfObjectsAndPredicates["birthdayboy"]["objects"])
# V2_TaskDescription_ = V2_TaskDescription_.replace("{List_of_predicates}", objectspredicates.ListOfObjectsAndPredicates["birthdayboy"]["predicates"])
V2_TaskDescription_ = V2_TaskDescription_.replace("{List_of_objects}", objectspredicates.ListOfObjectsAndPredicates["cowseatinggrass"]["objects"])
V2_TaskDescription_ = V2_TaskDescription_.replace("{List_of_predicates}", objectspredicates.ListOfObjectsAndPredicates["cowseatinggrass"]["predicates"])


v2_prompt = f"""
    {V2_TaskDescription_}
    """
# {ActualQuestion.Actual_instructions}

v1_prompt = f"""
    Your task is to generate scene graph for the provided video, to do so follow the steps below.

    1. Use the provided list to identify subjects/objects in the video. Only consider subjects/objects that appear both in the video and are on the list.
    2. Use the provided list of spatial predicates and action predicates that are happening in the video between objects.
    3. Finally, construct meaningful scene graph triplets in the format [subject-#id, predicate, object-#id] with a common sense, where the subject is performing the action (predicate), and the object is receiving the action. The #id refers to the unique identifier assigned to each object in the video.
    
    Important things to consider: 
        1. Ignore any objects in the video that are not on the list, as well as any objects on the list that do not appear in the video
        2. Track the objects in the consecutive frames to reassign the same #ids to the objects.
        3. Make sure the triplets constructed are logical and with common sense.

    {InContextExamples.InContext_Examples}

    Now from the given video generate scene graph
    Unique Objects in the video: [cattle, house, door, mountains]
    Unique Predicates: [stand behind, stand front,stand next to,walk next to,walk front]
    Unique Scene graph triplets in the video:
    """