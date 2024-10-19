import json
import os

list_data_dict = []
for data in ["/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_customchange/videochatgpt_tune_part0.json"]:
    data = json.load(open(data, "r"))
    for i in data:
        i['id'] = len(list_data_dict)
        list_data_dict.append(i)

print(list_data_dict[400])

# strList = "jump right,stand left,taller,jump past,jump behind,stand front,sit next to,sit behind,sit front,next to,front,stand next to,stand behind,walk right,walk next to,walk left,walk past,walk front,walk behind,faster,larger,stand with,stand right,walk with,walk toward,walk away,stop right,stop beneath,stand above,ride,run beneath,sit above,sit beneath,sit left,sit right,walk above,behind,watch,hold,feed,touch,right,left,follow,move front,move beneath,chase,run left,run right,lie next to,lie behind,play,move behind,jump beneath,fly with,fly past,move right,move left,swim front,swim left,move with,jump front,jump left,swim right,swim next to,jump next to,swim with,move past,bite,pull,jump toward,fight,run front,run behind,sit inside,drive,lie front,stop behind,lie left,stop left,lie right,creep behind,creep above,beneath,above,fall off,stop front,run away,run next to,away,jump away,fly next to,lie beneath,jump above,lie above,walk beneath,stand beneath,move toward,toward,past,move away,run past,fly behind,fly above,fly left,lie with,creep away,creep left,creep front,run with,run toward,creep right,creep past,fly front,fly right,fly away,fly toward,stop above,stand inside,kick,run above,swim beneath,jump with,lie inside,move above,move next to,creep next to,creep beneath,swim behind,stop next to,stop with,creep toward"
# listpred = strList.split(",")
# print(len(listpred))
# print(len(list(set(listpred))))