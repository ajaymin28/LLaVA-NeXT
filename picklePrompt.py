import pickle
from prompt_magic import Prompts
# import importlib
# importlib.reload(TaskDescription)
# importlib.reload(ActualQuestion)
# importlib.reload(Prompts)

# print(type(Prompts.v10_prompt), Prompts.v10_prompt)
Prompts_ = {
    "default": Prompts.v14_Task_description_AG_triplets_ZS,
    "version_14_AG_ZS_triplets": Prompts.v14_Task_description_AG_triplets_ZS,
    
    "version_14_AG_sgcls": Prompts.v14_Task_description_AG_triplets_sgcls,
    "version_14_vidor_triplets": Prompts.v14_prompt_vidor_triplets_withIds_without_obj_list,
    "version_14_vidor_quadruplets": Prompts.v14_prompt_vidor_quadruplets_withIds_without_obj_list,
    "version_14_vidvrd_without_obj_list": Prompts.v14_prompt_vidvrd_withIds_without_list,
    "version_14_vidvrd": Prompts.v14_prompt_vidvrd_withIds_with_list,
    "version_13_wids_temporal": Prompts.v13_prompt_withIds_temporal,
    "v13_prompt_vidvrd_withIds": Prompts.v13_prompt_vidvrd_withIds,
    "version_13_wids": Prompts.v13_prompt_withIds,
    "version_13_sam": Prompts.v13_prompt_sam,
    "version_13": Prompts.v13_prompt,
    "version_12": Prompts.v12_prompt,
    "version_11": Prompts.v11_prompt,
    "version_10": Prompts.v10_prompt,
    "version_9": Prompts.v9_prompt,
    "version_8": Prompts.v8_prompt,
    "version_7": Prompts.v7_prompt,
    "version_6": Prompts.v6_prompt,
    "version_3": Prompts.v3_prompt,
    "version_2": Prompts.v2_prompt,
    "version_1": Prompts.v1_prompt
}

with open("/home/jbhol/dso/gits/LLaVA-NeXT/prompts.pkl", "wb") as f:
    pickle.dump(Prompts_, f, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(Prompts,f)