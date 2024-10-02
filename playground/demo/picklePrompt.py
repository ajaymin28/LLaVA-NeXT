import pickle

from prompt_magic import TaskDescription, ActualQuestion, Prompts
import importlib
importlib.reload(TaskDescription)
importlib.reload(ActualQuestion)
importlib.reload(Prompts)

Detect_Count_objects = {
    "v0": "Identify the objects present in the video, assign each of them a unique integer #id for tracking.",
    "use_list_of_objects": {
        "v0":   """ 
                - For example, if in step-2 we get adult-1, and provided list given is [person, wood, trees] then replace adult-1 as person-1 since they do not change the meaning.
                - Another example, if in step-2 we get baby-2 and provided list given is [child, women, men] then replace baby-2 as child-1 since they do not change the meaning.
                """
    }
}


Prompts = {
    "default": Prompts.v6_prompt,
    "version_6": Prompts.v6_prompt,
    "version_3": Prompts.v3_prompt,
    "version_2": Prompts.v2_prompt,
    "version_1": Prompts.v1_prompt
}

with open("/home/jbhol/dso/gits/LLaVA-NeXT/playground/demo/prompts.pkl", "wb") as f:
    pickle.dump(Prompts, f, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(Prompts,f)