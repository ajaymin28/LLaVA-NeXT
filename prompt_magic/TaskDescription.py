
List_of_objects = []
List_of_predicates = [] 

vidvrd_predicates_numbered = """1.jump right 2.stand left 3.taller 4.jump past 5.jump behind 6.stand front 7.sit next to 8.sit behind 9.sit front 10.next to 11.front 12.stand next to 13.stand behind 14.walk right 15.walk next to 16.walk left 17.walk past 18.walk front 19.walk behind 20.faster 21.larger 22.stand with 23.stand right 24.walk with 25.walk toward 26.walk away 27.stop right 28.stop beneath 29.stand above 30.ride 31.run beneath 32.sit above 33.sit beneath 34.sit left 35.sit right 36.walk above 37.behind 38.watch 39.hold 40.feed 41.touch 42.right 43.left 44.follow 45.move front 46.move beneath 47.chase 48.run left 49.run right 50.lie next to 51.lie behind 52.play 53.move behind 54.jump beneath 55.fly with 56.fly past 57.move right 58.move left 59.swim front 60.swim left 61.move with 62.jump front 63.jump left 64.swim right 65.swim next to 66.jump next to 67.swim with 68.move past 69.bite 70.pull 71.jump toward 72.fight 73.run front 74.run behind 75.sit inside 76.drive 77.lie front 78.stop behind 79.lie left 80.stop left 81.lie right 82.creep behind 83.creep above 84.beneath 85.above 86.fall off 87.stop front 88.run away 89.run next to 90.away 91.jump away 92.fly next to 93.lie beneath 94.jump above 95.lie above 96.walk beneath 97.stand beneath 98.move toward 99.toward 100.past 101.move away 102.run past 103.fly behind 104.fly above 105.fly left 106.lie with 107.creep away 108.creep left 109.creep front 110.run with 111.run toward 112.creep right 113.creep past 114.fly front 115.fly right 116.fly away 117.fly toward 118.stop above 119.stand inside 120.kick 121.run above 122.swim beneath 123.jump with 124.lie inside 125.move above 126.move next to 127.creep next to 128.creep beneath 129.swim behind 130.stop next to 131.stop with 132.creep toward"""
vidvrd_objects_numbered = """1.antelope 2.person 3.dog 4.zebra 5.bicycle 6.horse 7.monkey 8.fox 9.elephant 10.lion 11.giant_panda 12.airplane 13.whale 14.watercraft 15.car 16.bird 17.cattle 18.rabbit 19.snake 20.frisbee 21.motorcycle 22.ball 23.domestic_cat 24.bear 25.red_panda 26.lizard 27.skateboard 28.sheep 29.squirrel 30.bus 31.sofa 32.train 33.turtle 34.tiger 35.hamster"""


# """
# Example: 
# If frame height,width is (480,280).
# and given bounding box pairs by frames : { 
# 'frame-0': [[[210.0, 267.3, 403.2, 267.3],[252.0, 8.1, 21.0, 8.1]]],
# 'frame-1': [[[210.0, 267.3, 403.2, 267.3],[210.0, 251.1, 222.6, 256.5]]], 
# 'frame-2': [[[210.0, 267.3, 403.2, 267.3],[252.0, 8.1, 21.0, 8.1]],[[210.0, 267.3, 403.2, 267.3],[210.0, 251.1, 222.6, 256.5]]]    
# }
# Answer:
# #sg_start
# { 
#   'frame-0': {
#     'objects': [[{'person-2': [210.0, 267.3, 403.2, 267.3]}, {'phone/camera-5': [252.0, 8.1, 21.0, 8.1]}]],
#     'relations':{
#         'attention': [['person-2','looking at','phone/camera-5']],
#         'contacting': [['person-2','holding','phone/camera-5']],
#         'spatial': [['phone/camera-5','in front of','person-2']]
#     }
#   },
#   'frame-1': {
#     'objects': [[{'person-2': [210.0, 267.3, 403.2, 267.3]}, {'pillow-7': [210.0, 251.1, 222.6, 256.5]}]],
#     'relations':{
#         'attention': [['person-2','not looking at','pillow-7']],
#         'contacting': [['person-2','holding','pillow-7']],
#         'spatial': [['pillow-7','in front of','person-2']]
#     }
#   },
#   'frame-2': { 
#     'objects': [[{'person-2': [210.0, 267.3, 403.2, 267.3]}, {'phone/camera-5': [252.0, 8.1, 21.0, 8.1]}], [{'person-2': [210.0, 267.3, 403.2, 267.3]}, {'pillow-7': [210.0, 251.1, 222.6, 256.5]}]],
#     'relations':{
#         'attention': [['person-2','not looking at','pillow-7'],['person-2','looking at','phone/camera-5']], 
#         'contacting': [['person-2','holding','pillow-7'],['person-2','holding','pillow-7']],
#         'spatial': [['pillow-7','in front of','person-2'],['phone/camera-5','in front of','person-2']]
#     }  
#   }
# }
# """

Task_description_v14_AG_with_list_GPT = """Your task is to create meaningful triplet which describe the provided video. the triplets consist of subject, relation, and object. 
The objects are: table,chair,bag,person,doorway,medicine,cup/glass/bottle,food,floor,broom,shoe,clothes,door,doorknob,groceries,closet/cabinet,laptop,bed,shelf,blanket,sandwich,refrigerator,vacuum,box,light,phone/camera,dish,paper/notebook,mirror,book,sofa/couch,television,pillow,towel,picture,window

There are three types of relationship: 1. attention 2. contacting and 3. spatial. 
Attention relation describes if person is looking at a specific object or not, or its unsure in the scene.
Contacting relation describes if person is interacting with object in the scene (e.g. holding, twisting). if there is not contacting, state 'not contacting'.
Spatial relation describes location of the object with respect to the person in the scene (e.g. behind, above)

The attention relations are: unsure, not looking at, looking at.
The contacting relations are: not contacting, sitting on, leaning on, other relationship, holding, touching, twisting, eating, drinking from, standing on,wearing,lying on,carrying,wiping,covered by,writing on,have it on the back.
The spatial relations are: in front of, beneath, behind, on the side of, in, above.

Step-1: List all unique objects present in the provided video using the objects list provided and assign random IDs to them for tracking. 
Step-2: Describe attention and contacting relationship for person and objects, and spatial relationship for object with respect to person.
Step-3: Provide triplets in the format <object-id,relation,object-id> using Step-1 and Step-2.

Example output: 
#sg_start 
    { 
        "objects": ["person-1", "table-5", "laptop-3"], 
        "triplets": [["person-1", "not looking at", "table-5"],["person-1", "looking at", "laptop-3"],["laptop-3", "in front of", "person-1"],["table-5", "on the side of", "person-1"],] 
    } 
#sg_end 
"""


Task_description_v14_ZS_AG_sgcls_short = """
You are provided with a video and a set of bounding box normlized coordinates pairs in format [xmin,ymin,xmax,ymax] with a specific frame ids of the video. 
A list of predefined objects is provided as follow: person,paper/notebook,picture,floor,blanket,window,vacuum,shoe,closet/cabinet,door,phone/camera,pillow,dish,towel,broom,doorknob,cup/glass/bottle,light,box,doorway,medicine,mirror,food,chair,book,television,sofa/couch,sandwich,bed,bag,laptop,table,clothes,groceries,shelf,refrigerator.

Your task is to detect object present in the given bounding box region from the predefined list of objects.


Given bounding box pairs by frames : { 
'frame-0': [[[0.5, 0.99, 0.96, 0.99],[0.6, 0.03, 0.05, 0.03]]],
'frame-1': [[[0.5, 0.99, 0.96, 0.99],[0.5, 0.93, 0.53, 0.95]]], 
'frame-2': [[[0.5, 0.99, 0.96, 0.99],[0.6, 0.03, 0.05, 0.03]],[[0.5, 0.99, 0.96, 0.99],[0.5, 0.93, 0.53, 0.95]]]    
}
Follow the output format:
#sg_start
{
    'frame-0': {
        'objects': [[{'person-2': [0.5, 0.99, 0.96, 0.99]}, {'phone/camera-5': [0.6, 0.03, 0.05, 0.03]}]],
        'relations':{
            'attention': [['person-2','looking at','phone/camera-5']],
            'contacting': [['person-2','holding','phone/camera-5']],
            'spatial': [['phone/camera-5','in front of','person-2']]
        }
    },
    'frame-1': {
        'objects': [[{'person-2': [0.5, 0.99, 0.96, 0.99]}, {'pillow-7': [0.5, 0.93, 0.53, 0.95]}]],
        'relations':{
            'attention': [['person-2','not looking at','pillow-7']],
            'contacting': [['person-2','holding','pillow-7']],
            'spatial': [['pillow-7','in front of','person-2']]
        }
    },
    'frame-2': { 
        'objects': [[{'person-2': [0.5, 0.99, 0.96, 0.99]}, {'phone/camera-5': [0.6, 0.03, 0.05, 0.03]}], [{'person-2': [0.5, 0.99, 0.96, 0.99]}, {'pillow-7': [0.5, 0.93, 0.53, 0.95]}]],
        'relations':{
            'attention': [['person-2','not looking at','pillow-7'],['person-2','looking at','phone/camera-5']], 
            'contacting': [['person-2','holding','pillow-7'],['person-2','holding','pillow-7']],
            'spatial': [['pillow-7','in front of','person-2'],['phone/camera-5','in front of','person-2']]
        }
    }
}
#sg_end
"""


Task_description_v14_ZS_AG_sgcls = """You are provided with a video and a set of bounding box coordinates pairs in format [xmin,ymin,xmax,ymax] for specific frames (identified by frame IDs). And a list of predefined objects as follows: person,paper/notebook,picture,floor,blanket,window,vacuum,shoe,closet/cabinet,door,phone/camera,pillow,dish,towel,broom,doorknob,cup/glass/bottle,light,box,doorway,medicine,mirror,food,chair,book,television,sofa/couch,sandwich,bed,bag,laptop,table,clothes,groceries,shelf,refrigerator.
Additionally, three types of predefined relationships are given to describe the interactions between the person and the objects across the frames.
1. Attention=looking at,unsure,not looking at. 
2. Spatial=behind,on the side of,in front of,above,beneath,in. 
3. Contacting=drinking from,leaning on,standing on,covered by,eating,wearing,touching,carrying,lying on,writing on,wiping,other relationship,holding,not contacting,twisting,have it on the back,sitting on.
Attention relationship indicates whether the person is visually focused on the object in the scene. Spatial relationship describes the object's position relative to the person within the scene. Contacting relationship specifies the physical interaction or contact between the person and the object.
Your task is as follows,
1. Detect object present in the given bounding box region from the predefined list of objects and assign a unique random #id to track the objects throughout the frames.
2. Identify relationship between objects and person from the predefined list. For each [person,object] pairs give Attention and Contatcting relations in form of triplets e.g. [person,relation,object] and for each [object,person] pairs, give spatial relation in form of triplets e.g. [object,realtion,person].

Example: 
Given bounding box pairs by frames : { 
'frame-0': [[[0.5, 0.99, 0.96, 0.99],[0.6, 0.03, 0.05, 0.03]]],
'frame-1': [[[0.5, 0.99, 0.96, 0.99],[0.5, 0.93, 0.53, 0.95]]], 
'frame-2': [[[0.5, 0.99, 0.96, 0.99],[0.6, 0.03, 0.05, 0.03]],[[0.5, 0.99, 0.96, 0.99],[0.5, 0.93, 0.53, 0.95]]]    
}
Follow the output format:
#sg_start
{ 
'frame-0': {
    'objects': {
        {'person-2': [0.5, 0.99, 0.96, 0.99]}, 
        {'phone/camera-5': [0.6, 0.03, 0.05, 0.03]}},
    'relations':{
        'attention': [['person-2','looking at','phone/camera-5']],
        'contacting': [['person-2','holding','phone/camera-5']],
        'spatial': [['phone/camera-5','in front of','person-2']]}},
'frame-1': {
    'objects': {
        {'person-2': [0.5, 0.99, 0.96, 0.99]}, 
        {'pillow-7': [0.5, 0.93, 0.53, 0.95]}},
    'relations':{
        'attention': [['person-2','not looking at','pillow-7']],
        'contacting': [['person-2','holding','pillow-7']],
        'spatial': [['pillow-7','in front of','person-2']]},},
'frame-2': { 
    'objects': {
        {'person-2': [0.5, 0.99, 0.96, 0.99]}, 
        {'phone/camera-5': [0.6, 0.03, 0.05, 0.03]},  
        {'pillow-7': [0.5, 0.93, 0.53, 0.95]}},
    'relations':{
        'attention': [['person-2','not looking at','pillow-7'],['person-2','looking at','phone/camera-5']], 
        'contacting': [['person-2','holding','pillow-7'],['person-2','holding','pillow-7']],
        'spatial': [['pillow-7','in front of','person-2'],['phone/camera-5','in front of','person-2']]}  
    }

}
#sg_end
"""

Task_description_v14_ZS_AG_predcls = """You are provided with a video and a set of [person,object] pairs, each associated with bounding box coordinates in format [xmin,ymin,xmax,ymax] for specific frames (identified by frame IDs). Additionally, three predefined types of relationships are given to describe the interactions between the person and the objects across the frames.

1. Attention=looking at,unsure,not looking at. 
2. Spatial=behind,on the side of,in front of,above,beneath,in. 
3. Contacting=drinking from,leaning on,standing on,covered by,eating,wearing,touching,carrying,lying on,writing on,wiping,other relationship,holding,not contacting,twisting,have it on the back,sitting on.

Attention relationship indicates whether the person is visually focused on the object in the scene. Spatial relationship describes the object's position relative to the person within the scene. Contacting relationship specifies the physical interaction or contact between the person and the object.

Your task is to identify relationships between [person,object] and [object,person] in the provided video from the predefined list of relations.

For each [person,object] pairs give Attention and Contatcting relations in form of triplets e.g. [person,relation,object] and 
for each [object,person] pairs, give spatial relation in form of triplets e.g. [object,realtion,person].

Example: 
Given: {'frame-0': [{'person': [0.5, 0.99, 0.96, 0.99]}, {'phone/camera': [0.6, 0.03, 0.05, 0.03]}],'frame-1': [{'person': [0.5, 0.99, 0.96, 0.99]}, {'pillow': [0.5, 0.93, 0.53, 0.95]}], 'frame-2': [{'person': [0.5, 0.99, 0.96, 0.99]}, {'phone/camera': [0.6, 0.03, 0.05, 0.03]},  {'pillow': [0.5, 0.93, 0.53, 0.95]}]}

Output:
#sg_start
{ 
'frame-0': {'attention': [['person','looking at','phone/camera']],'contacting': [['person','holding','phone/camera']],'spatial': [['phone/camera','in front of','person']]},
'frame-1': {'attention': [['person','not looking at','pillow']],'contacting': [['person','holding','pillow']],'spatial': [['pillow','in front of','person']]},
'frame-2': { 'attention': [['person','not looking at','pillow'],['person','looking at','phone/camera']], 'contacting': [['person','holding','pillow'],['person','holding','pillow']],'spatial': [['pillow','in front of','person'],['phone/camera','in front of','person']]}
}   
#sg_end
"""


# Task_video_description_v1 = f"""
#     The video containing 8 frames is provided.
#     The task is to provide frame by frame detailed description of the video which shows what is happening in the video in each frame.

#     In-context example 1:
#         #sg_start
#         {
#             "frame-1": "A child stands in the garden, holding a ball in their hand, while a dog waits nearby with a curious expression.",
#             "frame-2": "The child slowly raises their arm, preparing to throw the ball. The dog's attention is fully on the ball, poised to chase.",
#             "frame-3": "The child gently tosses the ball forward. The dog begins to move, anticipation in every step.",
#             "frame-4": "The ball bounces a few times on the grass, and the dog moves in for the catch, approaching with calculated steps.",
#             "frame-5": "The dog reaches the ball, takes it in its mouth, and looks back at the child, who is still standing in the same spot.",
#             "frame-6": "The child takes a few steps towards the dog, reaching out as if inviting the dog to return the ball.",
#             "frame-7": "The dog trots over to the child, holding the ball proudly, tail wagging slowly.",
#             "frame-8": "They both sit down on the grass, the dog still holding the ball, as the child leans over, sharing a quiet moment together."
#         }
#         #sg_end

#     In-context example 2:
#         #sg_start
#         {
#             "frame-1": "An adult is seated on a couch, holding a book, while a baby plays with a small toy on the floor in front of them.",
#             "frame-2": "The baby crawls slowly toward the adult, reaching out with tiny hands. The adult looks up from the book, smiling.",
#             "frame-3": "The adult puts down the book and extends their hand towards the baby, encouraging them to come closer.",
#             "frame-4": "The baby reaches the adult, who gently picks them up and sits them on their lap, taking a moment to adjust.",
#             "frame-5": "They sit quietly together for a moment, the adult pointing to pictures in the book and the baby watching attentively.",
#             "frame-6": "The baby touches the book's pages, exploring the textures with curious fingers, while the adult watches patiently.",
#             "frame-7": "The adult gently turns a page, showing the baby new images, speaking softly as they explain each picture.",
#             "frame-8": "They continue sharing the book, enjoying a peaceful, slow moment, the baby now fully focused on the pages."
#         }
#         #sg_end

    
#     Answer:
# """

# """
# The objects_entity lexicon containing 125 lexemes is numbered as follows:""" + opvsg_objects_numbered + """\n\
#     and relations_entity lexicon containing 56 lexemes is numbered as follows:""" +  opvsg_predicates_numbered + """\n\
    
#     Given the objects and relations lexeme, the task is to generate description followed by triplets from the generated description in the form of [objects_entity-id lexicon, relations_entity lexicon, objects_entity-id lexicon]. 
#     The id is randomly assigned to each object-entity to ensure uniqueness and tracking throughout the video.
#     Make sure the index of the objects_entity and relations_entity is maintained.
#     Select relations_entity which best describes the relation between two objects.

#     Note: It is possible that exact relations_entity or objects_entity might not be visible in the video, but those can be aligned, refer below examples.
#     example-1: The relations_entity "grasping" can be mapped to "20.holding".
#     example-2: The objects_entity "puppy" can be mapped to "43.dog".

#     In-context example 1:
#         #sg_start
#         {
#             "description" : "A child playing with ball on a street. He throws ball on the other side of the street, where another kid catches the ball.",
#             "triplets": [
#                 [["33.child-7", "38.playing with", "3.ball-2"], ["frame-1","frame-3"]],
#                 [["33.child-7", "51.throwing", "3.ball-2"], ["frame-2", "frame-5"]],
#                 [["33.child-5", "6.catching", "3.ball-2"], ["frame-5", "frame-8"]],
#                 [["33.child-5", "46.standing on", "118.ground"], ["frame-5", "frame-8"]]
#             ]
#         }
   
#         From the provided video which contains 8 frames, generate triplets:
#         Note: Please give the output in the format given above.

#         In-context example 2:
#         #sg_start
#         {
#             "description" : "An adult and a baby are in the kitchen, with the adult holding a bottle while feeding the baby who is sitting on a chair.",
#             "triplets": [
#                     [["0.adult-2", "20.holding", "15.bottle-4"], ["frame-1","frame-5"]],
#                     [["0.adult-2", "15.feeding", "1.baby-6"], ["frame-1","frame-5"]],
#                     [["1.baby-6", "45.sitting on", "32.chair"], ["frame-1","frame-5"]],
#                     [["1.baby-6", "12.drinking from", "15.bottle-4"], ["frame-1","frame-5"]]
#             ]
#         }
#         #sg_end
# """
Task_description_v13_with_ids_temporal = """
    The task is to generate description followed by triplets from the generated description in the form of [[subject-id, relation, object-id],[frame-start, frame-end]]. 
    The id is randomly assigned to each subject and object entity to ensure uniqueness and tracking throughout the video.
    The frame-start and frame-end denotes when activity starts and end in the video. 'start' and 'end' is an integer value between 0 to 7.

    Example 1:
        #sg_start
        {
            "description" : "A child playing with ball on a street. He throws ball on the other side of the street, where another kid catches the ball.",
            "triplets": [
                [["child-1", "playing with", "ball-3"],["frame-1", "frame-4"]],
                [["child-1", "throwing", "ball-3"],["frame-2","frame-4"]],
                [["child-2", "catching", "ball-3"],["frame-5", "frame-8"]],
                [["child-2", "standing on", "ground-0"]["frame-5","frame-8"]],
            ]
        }
        #sg_end
"""

Task_description_v13_with_ids = """
    The task is to generate description from the provided video followed by triplets formation using the generated description in the form of [subject-id, relation, object-id].
    The id is randomly assigned to each subject and object entity to ensure uniqueness and tracking throughout the video.

    Output Example 1:
        #sg_start
        {
            "description" : "A child playing with ball on a street. He throws ball on the other side of the street, where another kid catches the ball.",
            "triplets": [
                ["child-1", "playing with", "ball-3"],
                ["child-1", "throwing", "ball-3"],
                ["child-2", "catching", "ball-3"],
                ["child-2", "standing on", "ground-0"]
            ]
        }
        #sg_end
    Answer:
"""


opvsg_predicates_numbered = """0.beside 1.biting 2.blowing 3.brushing 4.caressing 5.carrying 6.catching 7.chasing 8.cleaning 9.closing 10.cooking 11.cutting 12.drinking from 13.eating 14.entering 15.feeding 16.grabbing 17.guiding 18.hanging from 19.hitting 20.holding 21.hugging 22.in 23.in front of 24.jumping from 25.jumping over 26.kicking 27.kissing 28.licking 29.lighting 30.looking at 31.lying on 32.next to 33.on 34.opening 35.over 36.picking 37.playing 38.playing with 39.pointing to 40.pulling 41.pushing 42.riding 43.running on 44.shaking hand with 45.sitting on 46.standing on 47.stepping on 48.stirring 49.swinging 50.talking to 51.throwing 52.touching 53.toward 54.walking on 55.watering 56.wearing"""
opvsg_objects_numbered = """0.adult 1.baby 2.bag 3.ball 4.ballon 5.basket 6.bat 7.bed 8.bench 9.beverage 10.bike 11.bird 12.blanket 13.board 14.book 15.bottle 16.bowl 17.box 18.bread 19.brush 20.bucket 21.cabinet 22.cake 23.camera 24.can 25.candle 26.car 27.card 28.carpet 29.cart 30.cat 31.cellphone 32.chair 33.child 34.chopstick 35.cloth 36.computer 37.condiment 38.cookie 39.countertop 40.cover 41.cup 42.curtain 43.dog 44.door 45.drawer 46.dustbin 47.egg 48.fan 49.faucet 50.fence 51.flower 52.fork 53.fridge 54.fruit 55.gift 56.glass 57.glasses 58.glove 59.grain 60.guitar 61.hat 62.helmet 63.horse 64.iron 65.knife 66.light 67.lighter 68.mat 69.meat 70.microphone 71.microwave 72.mop 73.net 74.noodle 75.others 76.oven 77.pan 78.paper 79.piano 80.pillow 81.pizza 82.plant 83.plate 84.pot 85.powder 86.rack 87.racket 88.rag 89.ring 90.scissor 91.shelf 92.shoe 93.simmering 94.sink 95.slide 96.sofa 97.spatula 98.sponge 99.spoon 100.spray 101.stairs 102.stand 103.stove 104.switch 105.table 106.teapot 107.towel 108.toy 109.tray 110.tv 111.vaccum 112.vegetable 113.washer 114.window 115.ceiling 116.floor 117.grass 118.ground 119.rock 120.sand 121.sky 122.snow 123.tree 124.wall 125.water"""




vidvrd_predicates_numbered = """1.jump right 2.stand left 3.taller 4.jump past 5.jump behind 6.stand front 7.sit next to 8.sit behind 9.sit front 
10.next to 11.front 12.stand next to 13.stand behind 14.walk right 15.walk next to 16.walk left 17.walk past 18.walk front 19.walk behind 20.faster 
21.larger 22.stand with 23.stand right 24.walk with 25.walk toward 26.walk away 27.stop right 28.stop beneath 29.stand above 30.ride 31.run beneath 
32.sit above 33.sit beneath 34.sit left 35.sit right 36.walk above 37.behind 38.watch 39.hold 40.feed 41.touch 42.right 43.left 44.follow 45.move front
 46.move beneath 47.chase 48.run left 49.run right 50.lie next to 51.lie behind 52.play 53.move behind 54.jump beneath 55.fly with 56.fly past 
 57.move right 58.move left 59.swim front 60.swim left 61.move with 62.jump front 63.jump left 64.swim right 65.swim next to 66.jump next to 
 67.swim with 68.move past 69.bite 70.pull 71.jump toward 72.fight 73.run front 74.run behind 75.sit inside 76.drive 77.lie front 78.stop behind 
 79.lie left 80.stop left 81.lie right 82.creep behind 83.creep above 84.beneath 85.above 86.fall off 87.stop front 88.run away 89.run next to 90.away 
 91.jump away 92.fly next to 93.lie beneath 94.jump above 95.lie above 96.walk beneath 97.stand beneath 98.move toward 99.toward 100.past 101.move away 
 102.run past 103.fly behind 104.fly above 105.fly left 106.lie with 107.creep away 108.creep left 109.creep front 110.run with 111.run toward 
 112.creep right 113.creep past 114.fly front 115.fly right 116.fly away 117.fly toward 118.stop above 119.stand inside 120.kick 121.run above 
 122.swim beneath 123.jump with 124.lie inside 125.move above 126.move next to 127.creep next to 128.creep beneath 129.swim behind 130.stop next to 
 131.stop with 132.creep toward"""

vidvrd_objects_numbered = """1.antelope 2.person 3.dog 4.zebra 5.bicycle 6.horse 7.monkey 8.fox 9.elephant 10.lion 11.giant_panda 12.airplane 13.whale 
14.watercraft 15.car 16.bird 17.cattle 18.rabbit 19.snake 20.frisbee 21.motorcycle 22.ball 23.domestic_cat 24.bear 25.red_panda 26.lizard 27.skateboard 
28.sheep 29.squirrel 30.bus 31.sofa 32.train 33.turtle 34.tiger 35.hamster"""


vidvrd_action_preds = """1.bite 2.chase 3.creep 4.drive 5.fall off 6.faster 7.feed 8.fight 9.fly 10.fly with 11.follow 12.hold 13.jump 14.jump with 15.kick 16.larger 17.lie 18.lie inside 19.lie with 20.move 21.move with 22.play 23.pull 24.ride 25.run 26.run with 27.sit 28.sit inside 29.stand 30.stand inside 31.stand with 32.stop 33.stop with 34.swim 35.swim with 36.taller 37.touch 38.walk 39.walk with 40.watch"""
vidvrd_sptial_preds = """1.toward 2.past 3.above 4.left 5.beneath 6.right 7.behind 8.away 9.front 10.next to"""

Task_description_v14_vidvrd_without_list = f"""From the provided video, the task is to create meaningful action-based video scene-graph quadruplets.
You are given predefined objects list ["""+ vidvrd_objects_numbered + """], action predicates list ["""+ vidvrd_action_preds + """] and sptial predicates list ["""+ vidvrd_sptial_preds + """].

Follow the steps below to create quadruplets.
1. Create description of the provided video using provided list of objects and predicates list.
2. Generate meaningful scene-graph quadruplets from the obtained description from step-1. 

Example: 
    #sg_start
    {
        "description": "There are two lions in the scene. The smaller lion follow behind larger lion and the larger lion walk front of the smaller lion. The smaller lion stand behind the larger lion. The larger lion walk past the smaller lion. The larger lion walk next to the smaller lion for a moment.",
        "triplets": [
            ["10.lion-3","11.follow","7.behind","10.lion-1"],
            ["10.lion-1","38.walk","9.front","10.lion-3"],
            ["10.lion-3","stand","7.behind","10.lion-1"],
            ["10.lion-1","38.walk","2.past","10.lion-3"],
            ["10.lion-1","38.walk","10.next to","10.lion-3"]
        ]
    }
    #sg_end

Response:
"""
Task_description_v14_vidvrd_with_list_GPT_wo_obj_list = """Your task is to create meaningful quadruplets which describe the provided video. Quadruplets consist of subject, action relation, spatial relation, and object. The relationship consists of two entities: 1. action and 2. spatial. The action relations are: bite,chase,creep,drive,fall off,faster,feed,fight,fly,fly with,follow,hold,jump,jump with,kick,larger,lie,lie inside,lie with,move,move with,play,pull,ride,run,run with,sit,sit inside,stand,stand inside,stand with,stop,stop with,swim,swim with,taller,touch,walk,walk with,watch. The spatial relations are: toward,past,above,left,beneath,right,behind,away,front,next to. Step-1: List all unique objects present in the provided video using the objects list provided and assign random IDs to them for tracking. Step-2: Describe all possible relationships between objects obtained in Step-1 using the relationship lists provided. Ensure every spatial and action relation between each pair is captured in a detailed manner. For each object pair, provide a list of multiple spatial relations (e.g., front, left, right, etc.). Step-3: Provide quadruplets in the format <object-id, action, spatial, object-id> using Step-1 and Step-2, ensuring all possible spatial and action relations are described for each pair of objects. Example output: #sg_start { "objects": ["person-1", "person-2", "dog-3"], "quadruplets": [ ["person-1", "sit", "front", "person-2"],["person-1", "sit", "left", "person-2"],["person-1", "sit", "beneath", "person-2"],["person-1", "sit", "behind", "person-2"],["person-1", "sit", "next to", "dog-3"],["dog-3", "sit", "front", "person-2"],["person-2", "sit", "above", "dog-3"]] } #sg_end """
Task_description_v14_vidvrd_with_list_GPT = """Your task is to create meaningful quadruplets which describe the provided video. Quadruplets consist of subject, action relation, spatial relation, and object. The objects are: bus,motorcycle,sheep,frisbee,antelope,turtle,lizard,airplane,train,horse,giant_panda,bird,hamster,elephant,ball,watercraft,red_panda,rabbit,bear,whale,domestic_cat,lion,bicycle,monkey,squirrel,sofa,snake,car,tiger,fox,skateboard,zebra,dog,cattle,person. The relationship consists of two entities: 1. action and 2. spatial. The action relations are: bite,chase,creep,drive,fall off,faster,feed,fight,fly,fly with,follow,hold,jump,jump with,kick,larger,lie,lie inside,lie with,move,move with,play,pull,ride,run,run with,sit,sit inside,stand,stand inside,stand with,stop,stop with,swim,swim with,taller,touch,walk,walk with,watch. The spatial relations are: toward,past,above,left,beneath,right,behind,away,front,next to. Step-1: List all unique objects present in the provided video using the objects list provided and assign random IDs to them for tracking. Step-2: Describe all possible relationships between objects obtained in Step-1 using the relationship lists provided. Ensure every spatial and action relation between each pair is captured in a detailed manner. For each object pair, provide a list of multiple spatial relations (e.g., front, left, right, etc.). Step-3: Provide quadruplets in the format <object-id, action, spatial, object-id> using Step-1 and Step-2, ensuring all possible spatial and action relations are described for each pair of objects. Example output: #sg_start { "objects": ["person-1", "person-2", "dog-3"], "quadruplets": [ ["person-1", "sit", "front", "person-2"],["person-1", "sit", "left", "person-2"],["person-1", "sit", "beneath", "person-2"],["person-1", "sit", "behind", "person-2"],["person-1", "sit", "next to", "dog-3"],["dog-3", "sit", "front", "person-2"],["person-2", "sit", "above", "dog-3"]] } #sg_end """



Task_description_v14_vidvrd_without_list_Old = f"""From the provided video, create meaningful action-based video scene-graph quadruplets by first providing a detailed description in 100-200 words with the focus on the main objects and activities in the video (e.g., objects present, their placements with other main objects, and actions). Then convert this description into ["object-id", "action predicate","spatial predicate", "object-id"] quadruplets and assign unique IDs to objects for tracking.
Note:
    - Quadruplets should be strictly four entities only (e.g. ["object-id","action predicate","spatial predicate", "object-id"]).
    - The output format should be consistant as shown in below example.
    - The objects can be one of the following ["""+ vidvrd_objects_numbered + """]
    - The action predicates can be one of the following ["""+ vidvrd_action_preds + """]
    - The spatial predicates can be one of the following ["""+ vidvrd_sptial_preds + """].
    - Focus on the main objects when creating Quadruplets.

    Example: 
    #sg_start
    {
        "description": "The larger lion leading and the smaller lion follow behind. The larger lion walk front of the smaller lion. The smaller lion stand behind the larger lion. The larger lion walk past the smaller lion. The larger lion walk next to the smaller lion for a moment.",
        "triplets": [["lion-3","follow","behind","lion-1"],["lion-1","walk","front","lion-3"],["lion-3","stand","behind","lion-1"],["lion-1","walk","past","lion-3"],["lion-1","walk","next to","lion-3"]]
    }
    #sg_end
    
    Response:
"""

## 1 vid
Task_description_v13_vidvrd_sam_with_list = """The objects lexicon containing 35 lexemes is numbered as follows:""" + vidvrd_objects_numbered + """ 
and relations lexicon containing 132 lexemes is numbered as follows:""" +  vidvrd_predicates_numbered + """.
From the provided video, create meaningful action-based video scene-graph triplets by first providing a detailed description in 100-200 words using predefined lexicon with the focus on the main objects and activities in the video (e.g., objects present, their placements, and actions). Then convert this description into [subject-id, relation, object-id] triplets using predefined lexicon, assign unique IDs to subjects and objects for tracking.
Note:
    - Triplets should be only three entities (e.g. ["2.person-3", "12.stand next to", "3.dog-5"])
    - The output format should be consistant as shown in below example.

    Example: 
    #sg_start
    {
        "description": "The 21.larger 10.lion leading and the smaller lion follow behind. 
        The larger lion walk front of the smaller lion. 
        The smaller lion stand behind the larger lion. 
        The larger lion walk past the smaller lion. 
        The larger lion walk next to the smaller lion for a moment.",

        "triplets": [
            ["10.lion-3", "44.follow behind", "10.lion-1"],
            ["10.lion-1", "18.walk front", "10.lion-3"],
            ["10.lion-3", "13.stand behind", "10.lion-1"],
            ["10.lion-1", "17.walk past", "10.lion-3"],
            ["10.lion-1", "15.walk next to", "10.lion-3"]
        ]
    }
    #sg_end
    
    Response:
"""

# instead of quadruplets its triplets
# The objects are: condiment,simmering,knife,candle,shoe,basket,fan,oven,bottle,fork,wall,curtain,fence,rack,blanket,cloth,flower,guitar,vaccum,baby,child,drawer,chopstick,table,iron,book,pan,card,window,cookie,pot,carpet,camera,countertop,bucket,cake,powder,rock,sofa,sand,stand,plant,water,beverage,cart,cabinet,grass,stove,noodle,piano,switch,glasses,computer,light,mat,washer,meat,tv,pizza,paper,cup,cover,lighter,board,ballon,spoon,box,cellphone,towel,ground,adult,bed,faucet,bat,rag,tree,scissor,chair,bike,glass,plate,can,snow,egg,toy,ring,mop,racket,grain,vegetable,spray,stairs,helmet,pillow,car,bread,door,sink,gift,horse,net,floor,fridge,others,brush,cat,sky,ceiling,tray,fruit,ball,dog,microwave,spatula,hat,shelf,bench,bowl,sponge,teapot,dustbin,microphone,slide,bag,bird,glove.
Task_description_v14_vidor_triplets_with_list_GPT = """Your task is to create meaningful triplets which describes the provided video. triplets consist of subject, relation, and object.  The relations are: sitting on,hugging,blowing,biting,closing,catching,carrying,swinging,hitting,stirring,touching,toward,pulling,kissing,walking on,feeding,hanging from,over,cleaning,playing,kicking,on,cooking,shaking hand with,guiding,licking,in,pushing,next to,in front of,playing with,jumping over,standing on,pointing to,talking to,lying on,grabbing,opening,caressing,cutting,drinking from,jumping from,watering,chasing,throwing,riding,eating,holding,running on,brushing,beside,entering,picking,wearing,stepping on,lighting,looking at. Step-1: List all unique objects present in the provided video using the objects list provided and assign random IDs to them for tracking.  Step-2: Describe all possible relationships between objects obtained in Step-1 using the relations lists provided. Ensure every relation between each pair is captured in a detailed manner. For each object pair, provide a list of multiple relations (e.g., in front of, sitting on, next to, etc.). Step-3: Provide triplets in the format [object-id, relation, object-id] using Step-1 and Step-2, ensuring all possible relations are described for each pair of objects. Example output: #sg_start { "objects": ["child-1", "ground-0", "car-6","ball-5"],"triplets": [["child-1", "standing on", "ground-0"],["child-1", "holding", "ball-5"],["child-1", "throwing", "ball"],["car-6", "on", "ground-0"],["child-4", "catching" "ball-5"]] } #sg_end """

## The objects are: condiment,simmering,knife,candle,shoe,basket,fan,oven,bottle,fork,wall,curtain,fence,rack,blanket,cloth,flower,guitar,vaccum,baby,child,drawer,chopstick,table,iron,book,pan,card,window,cookie,pot,carpet,camera,countertop,bucket,cake,powder,rock,sofa,sand,stand,plant,water,beverage,cart,cabinet,grass,stove,noodle,piano,switch,glasses,computer,light,mat,washer,meat,tv,pizza,paper,cup,cover,lighter,board,ballon,spoon,box,cellphone,towel,ground,adult,bed,faucet,bat,rag,tree,scissor,chair,bike,glass,plate,can,snow,egg,toy,ring,mop,racket,grain,vegetable,spray,stairs,helmet,pillow,car,bread,door,sink,gift,horse,net,floor,fridge,others,brush,cat,sky,ceiling,tray,fruit,ball,dog,microwave,spatula,hat,shelf,bench,bowl,sponge,teapot,dustbin,microphone,slide,bag,bird,glove.
Task_description_v14_vidor_quadruplets_with_list_GPT = """Your task is to create meaningful quadruplets which describes the provided video. Quadruplets consist of subject, action relation, spatial relation, and object.  The relationship consists of two entities: 1. action and 2. spatial. The action relations are: sitting,hugging,blowing,biting,closing,catching,carrying,swinging,hitting,stirring,touching,pulling,kissing,walking,feeding,hanging,cleaning,playing,kicking,cooking,shaking hand,guiding,licking,pushing,playing,jumping,standing,pointing,talking,lying,grabbing,opening,caressing,cutting,drinking,jumping,watering,chasing,throwing,riding,eating,holding,running,brushing,entering,picking,wearing,stepping,lighting,looking. The spatial relations are: on,toward,past,above,left,beneath,right,behind,away,front,over,in,beside,next to,with,in front of,to,from,at,None. Step-1: List all unique non repeating objects present in the provided video using the objects list provided and assign random IDs to them for tracking. Step-2: Describe all possible relationships between objects obtained in Step-1 using the relationship lists provided. Ensure every spatial and action relation between each pair is captured in a detailed manner. For each object pair, provide a list of multiple spatial relations (e.g., front, left, right, etc.). Step-3: Provide quadruplets in the format [subject-id, action relation, spatial relation, object-id] using Step-1 and Step-2, ensuring all possible spatial and action relations are described for each pair of objects. Example output: #sg_start { "objects": ["child-1", "ground-0", "car-6","ball-5"],"quadruplets": [ ["child-1", "standing","on", "ground-0"],["child-1", "holding","None", "ball-5"],["child-1", "throwing", "None", "ball"],["car-6", "None","on", "ground-0"],["child-4", "catching","None" "ball-5"]] } #sg_end """

Task_description_v14_vidor_with_list_GPT = """Your task is to create meaningful quadruplets which describe the provided video. Quadruplets consist of subject, action relation, spatial relation, and object. The objects are: bus,motorcycle,sheep,frisbee,antelope,turtle,lizard,airplane,train,horse,giant_panda,bird,hamster,elephant,ball,watercraft,red_panda,rabbit,bear,whale,domestic_cat,lion,bicycle,monkey,squirrel,sofa,snake,car,tiger,fox,skateboard,zebra,dog,cattle,person. The relationship consists of two entities: 1. action and 2. spatial. The action relations are: bite,chase,creep,drive,fall off,faster,feed,fight,fly,fly with,follow,hold,jump,jump with,kick,larger,lie,lie inside,lie with,move,move with,play,pull,ride,run,run with,sit,sit inside,stand,stand inside,stand with,stop,stop with,swim,swim with,taller,touch,walk,walk with,watch. The spatial relations are: toward,past,above,left,beneath,right,behind,away,front,next to. Step-1: List all unique objects present in the provided video using the objects list provided and assign random IDs to them for tracking. Step-2: Describe all possible relationships between objects obtained in Step-1 using the relationship lists provided. Ensure every spatial and action relation between each pair is captured in a detailed manner. For each object pair, provide a list of multiple spatial relations (e.g., front, left, right, etc.). Step-3: Provide quadruplets in the format <object-id, action, spatial, object-id> using Step-1 and Step-2, ensuring all possible spatial and action relations are described for each pair of objects. Example output: #sg_start { "objects": ["person-1", "person-2", "dog-3"], "quadruplets": [ ["person-1", "sit", "front", "person-2"],["person-1", "sit", "left", "person-2"],["person-1", "sit", "beneath", "person-2"],["person-1", "sit", "behind", "person-2"],["person-1", "sit", "next to", "dog-3"],["dog-3", "sit", "front", "person-2"],["person-2", "sit", "above", "dog-3"]] } #sg_end """


## 1 vid
Task_description_v13_sam_with_list = """The objects lexicon containing 125 lexemes is numbered as follows:""" + opvsg_objects_numbered + """ 
and relations lexicon containing 56 lexemes is numbered as follows:""" +  opvsg_predicates_numbered + """.
Create action-based video scene-graph triplets by first providing a detailed description in 100-200 words using predefined lexicon with the focus on the main activities in the video (e.g., objects present, their placements, and actions). Then convert this description into [subject-id, relation, object-id] triplets using predefined lexicon, assign unique IDs to subjects and objects for tracking.
Note:
    - Triplets should be strictly three entities (i.e. subject-id, relation, object-id)
    - The output format should be consistant as shown in below example.

    Example: 
    #sg_start
    {
        "description": "A child stands on a ground, holding a ball. The child starts throwing it across to his child friend, who steps forward, arms ready. A car slows down, passing carefully behind as the second child catches the ball with a grin",
        "triplets": [
            ["33.child-1", "46.standing on", "118.ground-0"],
            ["33.child-1", "20.holding", "3.ball-5"],
            ["33.child-1", "51.throwing", "3.ball-5"],
            ["26.car-6", "33.on", "118.ground-0"],
            ["33.child-4", "6.catching", "3.ball-5"]
        ]
    }
    #sg_end
    
    Response:
"""


## 1 vid # GPT improved/shorten
Task_description_v13_sam = """Create action-based video scene-graph triplets by first providing a detailed description of the main activities in the video (e.g., objects present, their placements, and actions). Convert this description into [subject-id, relation, object-id] triplets, assigning unique IDs to subjects and objects for tracking.
Note:
    - Triplets should be strictly three entities (i.e. subject-id, relation, object-id)
Example: 
#sg_start
{
    "description": "A boy stands on a quiet street, holding a ball. He throws it across to his friend, who steps forward, arms ready. A car slows down, passing carefully behind as the second child catches the ball with a grin.",
    "triplets": [
        ["child-1", "standing on", "street-0"],
        ["child-1", "holding", "ball-3"],
        ["child-1", "throws", "ball-3"],
        ["car-6", "on", "street-0"],
        ["child-4", "catches", "ball-3"]
    ]
}
#sg_end
Response:#sg_start
"""

# ## 1 vid
# Task_description_v13_sam = """Generate action-based video scene-graph triplets by first describing the video in detail, then convert this description into [subject, relation, object] triplets.
# Note:
#     - The video description should focus on the main activities (e.g. what objects are present in the video? how they are placed in the scene? what they are doing? And what actions they are performing?)
#     - Triplets should be strictly three entities (i.e. subject, relation, object)
# Example: 
# #sg_start
# {
#     "description": "A boy stands on a quiet street, holding a ball. He throws it across to his friend, who steps forward, arms ready. A car slows down, passing carefully behind as the second child catches the ball with a grin.",
#     "triplets": [
#         ["child", "standing on", "street"],
#         ["child", "holding", "ball"],
#         ["child", "throws", "ball"],
#         ["car", "on", "street"],
#         ["child", "catches", "ball"]
#     ]
# }
# #sg_end
# Response:#sg_start
# """

# 1 vid 0.01
# Task_description_v13_sam = """
#         Generate action-based video scene-graph triplets by first describing the video in detail focusing on the main actors and objects as well as relations between them in the video, then convert this description into [subject, relation, object] triplets.
#         Example: 
#         #sg_start
#         {
#             "description": "A boy stands on a quiet street, holding a ball. He throws it across to his friend, who steps forward, arms ready. A car slows down, passing carefully behind as the second child catches the ball with a grin.",
#             "triplets": [
#                 ["child", "standing on", "street"],
#                 ["child", "holding", "ball"],
#                 ["child", "throws", "ball"],
#                 ["car", "on", "street"],
#                 ["child", "catches", "ball"]
#             ]
#         }
#         #sg_end
#         Response:#sg_start
# """

Task_description_v13 = """
    The task is to generate description from the provided video followed by triplets formation using the generated description in the form of [subject, relation, object]. 

    Output Example 1:
        #sg_start
        {
            "description" : "A child playing with ball on a street. He throws ball on the other side of the street, where another kid catches the ball.",
            "triplets": [
                ["child", "playing with", "ball"],
                ["child", "throwing", "ball"],
                ["child", "catching", "ball"],
                ["child", "standing on", "ground"]
            ]
        }
        #sg_end
    Answer:
"""

Task_description_v12 = f"""The objects_entity lexicon containing 125 lexemes is numbered as follows:""" + opvsg_objects_numbered + """\n\
    and relations_entity lexicon containing 56 lexemes is numbered as follows:""" +  opvsg_predicates_numbered + """\n\
    
    Given the objects and relations lexeme, the task is to generate triplets from the provided video in the form of [objects_entity-id lexicon, relations_entity lexicon, objects_entity-id lexicon]. 
    The id is randomly assigned to each object-entity to ensure uniqueness and tracking throughout the video.
    Make sure the index of the objects_entity and relations_entity is maintained.
    Select relations_entity which best describes the relation between two objects.

    Note: It is possible that exact relations_entity or objects_entity might not be visible in the video, but those can be aligned, refer below examples.
    example-1: The relations_entity "grasping" can be mapped to "20.holding".
    example-2: The objects_entity "puppy" can be mapped to "43.dog".
    example-3: The relations_entity "adjacent to" can be mapped to "32.next to".
    example-4: The relations_entity "leaping over" can be mapped to "25.jumping over".
    example-5: The objects_entity "kid" can be mapped to "33.child".

    In-context example 1:
        #sg_start
        {
        "description" : "A child and a dog enjoy a playful day together on the grass. The child throws a ball, and the dog eagerly catches it, leading to a lively game of chase. After some energetic play, including the dog jumping over the child, they sit down to relax, eventually lying side by side on the grass, peacefully looking up at the sky together."
        "triplets": [
            [["33.child-7", "20.holding", "3.ball-2"], [frame-1]],
            [["43.dog-3", "24.jumping from", "33.child-7"], [frame-1]],
            [["43.dog-3", "0.beside", "33.child-7"], [frame-1, frame-8]],
            [["33.child-7", "51.throwing", "3.ball-2"], [frame-2]],
            [["43.dog-3", "6.catching", "3.ball-2"], [frame-2]],
            [["33.child-7", "7.chasing", "43.dog-3"], [frame-3]],
            [["43.dog-3", "43.running on", "117.grass-5"], [frame-3]],`
            [["33.child-7", "36.picking", "118.ground-9"], [frame-4]],
            [["43.dog-3", "30.looking at", "33.child-7"], [frame-4, frame-5]],
            [["43.dog-3", "25.jumping over", "33.child-7"], [frame-5]],
            [["33.child-7", "45.sitting on", "117.grass-5"], [frame-6]],
            [["43.dog-3", "45.sitting on", "117.grass-5"], [frame-6]],
            [["33.child-7", "50.talking to", "43.dog-3"], [frame-6]],
            [["33.child-7", "31.lying on", "117.grass-5"], [frame-7]],
            [["43.dog-3", "31.lying on", "117.grass-5"], [frame-7]],
            [["43.dog-3", "32.next to", "33.child-7"], [frame-7]],
            [["33.child-7", "30.looking at", "121.sky-4"], [frame-8]],
            [["43.dog-3", "30.looking at", "121.sky-4"], [frame-8]]
            ],
        }  

    In-context example 2:
        #sg_start
        {
            "scene": { 
                "description" : "An adult and a baby are in the kitchen, with the adult holding a bottle while feeding the baby who is sitting on a chair.",
            },
            "triplets": [
                [["0.adult-2", "20.holding", "15.bottle-4"], [frame-1,frame-5]],
                [["0.adult-2", "15.feeding", "1.baby-6"], [frame-1]],
                [["1.baby-6", "45.sitting on", "32.chair"], [frame-1]],
                [["1.baby-6", "23.in front of", "0.adult-2"],[frame-1]],
                [["1.baby-6", "12.drinking from", "15.bottle-4"], [frame-2,frame-4]],
                [["0.adult-2", "30.looking at", "1.baby-6"],[frame-2,frame-5]],
                [["1.baby-6", "20.holding", "15.bottle-4"], [frame-3]],
                [["0.adult-2", "53.toward", "1.baby-6"],[frame-3]],
                [["0.adult-2", "4.caressing", "1.baby-6"], [frame-4]],
                [["1.baby-6", "30.looking at", "0.adult-2"], [frame-5,frame-7]],
                [["0.adult-2", "47.stepping on", "116.floor-0"], [frame-6]]
                [["1.baby-6", "31.lying on", "116.floor-0"], [frame-6]]
                [["1.baby-6", "0.beside", "0.adult-2"],[frame-6]]
                [["0.adult-2", "36.picking", "108.toy-9"], [frame-7]],
                [["0.adult-2", "38.playing with", "108.toy-9"], [frame-8]]
                [["1.baby-6", "38.playing with", "108.toy-9"], [frame-8]]
                [["0.adult-2", "45.sitting on", "116.floor-0"], [frame-8]]
                [["1.baby-6", "45.sitting on", "116.floor-0"],[frame-8]]
            ]
        }
        #sg_end
   
        From the provided video which contains 8 frames, generate triplets:
        Note: Please give the output in the format given above.
"""


Task_description_v11 = f"""The objects_entity lexicon containing 125 lexemes is numbered as follows:""" + opvsg_objects_numbered + """\n\
    and relations_entity lexicon containing 56 lexemes is numbered as follows:""" +  opvsg_predicates_numbered + """\n\
    
    Given the objects and relations lexeme, the task is to generate triplets from the provided video in the form of [objects_entity-id lexicon, relations_entity lexicon, objects_entity-id lexicon]. 
    The id is randomly assigned to each object-entity to ensure uniqueness and tracking throughout the video.
    Make sure the index of the objects_entity and relations_entity is maintained.
    Select relations_entity which best describes the relation between two objects.

    Note: It is possible that exact relations_entity or objects_entity might not be visible in the video, but those can be aligned, refer below examples.
    example-1: The relations_entity "grasping" can be mapped to "20.holding".
    example-2: The objects_entity "puppy" can be mapped to "43.dog".
    example-3: The relations_entity "adjacent to" can be mapped to "32.next to".
    example-4: The relations_entity "leaping over" can be mapped to "25.jumping over".
    example-5: The objects_entity "kid" can be mapped to "33.child".

    In-context example 1:
        #sg_start
        {
            "scene": { 
                "description" : "A child and a dog play outside, with the child holding a ball while the dog jumps beside them on the grass.",
            },
            "frame-1": {
                "descriptions": ["The child holds the ball while the dog jumps beside them.", "The dog is next to the child on the grass."],
                "triplets": [["33.child-7", "20.holding", "3.ball-2"], ["43.dog-3", "24.jumping from", "33.child-7"], ["43.dog-3", "0.beside", "33.child-7"]]
            },  
            "frame-2": {
                "descriptions": ["The child throws the ball to the dog.", "The dog catches the ball mid-air."],
                "triplets": [["33.child-7", "51.throwing", "3.ball-2"], ["43.dog-3", "6.catching", "3.ball-2"]]
            },  
            "frame-3": {
                "descriptions": ["The child chases the dog.", "The dog runs away with the ball."],
                "triplets": [["33.child-7", "7.chasing", "43.dog-3"], ["43.dog-3", "43.running on", "117.grass-5"]]
            },
            "frame-4": {
                "descriptions": ["The child picks up a stick to throw it to the dog.", "The dog watches intently."],
                "triplets": [["33.child-7", "36.picking", "118.ground-9"], ["43.dog-3", "30.looking at", "33.child-7"]]
            },
            "frame-5": {
                "descriptions": ["The dog jumps over the child.", "The child laughs, watching the dog."],
                "triplets": [["43.dog-3", "25.jumping over", "33.child-7"], ["33.child-7", "30.looking at", "43.dog-3"]]
            },
            "frame-6": {
                "descriptions": ["The child and the dog sit on the grass together.", "They look at each other contentedly."],
                "triplets": [["33.child-7", "45.sitting on", "117.grass-5"], ["43.dog-3", "45.sitting on", "117.grass-5"], ["33.child-7", "50.talking to", "43.dog-3"]]
            },
            "frame-7": {
                "descriptions": ["The child and dog lie down together on the grass.", "The dog lies beside the child."],
                "triplets": [["33.child-7", "31.lying on", "117.grass-5"], ["43.dog-3", "31.lying on", "117.grass-5"], ["43.dog-3", "32.next to", "33.child-7"]]
            },
            "frame-8": {
                "descriptions": ["The child and dog rest quietly, side by side.", "They both look up at the sky."],
                "triplets": [["33.child-7", "0.beside", "43.dog-3"], ["43.dog-3", "30.looking at", "121.sky-4"], ["33.child-7", "30.looking at", "121.sky-4"]]
            },
            "st progression": "The scene depicts a playful interaction between a child and a dog. The child initially holds a ball, and the dog eagerly jumps around. As they play, the child throws the ball, and the dog catches it. They continue playing and eventually rest together on the grass, enjoying each other's company."
        }
        #sg_end

    In-context example 2:
        #sg_start
        {
            "scene": { 
                "description" : "An adult and a baby are in the kitchen, with the adult holding a bottle while feeding the baby who is sitting on a chair.",
            },
            "frame-1": {
                "descriptions": ["The adult holds the bottle while feeding the baby.", "The baby is sitting on the chair in front of the adult."],
                "triplets": [ ["0.adult-2", "20.holding", "15.bottle-4"], ["0.adult-2", "15.feeding", "1.baby-6"], ["1.baby-6", "45.sitting on", "32.chair"], ["1.baby-6", "23.in front of", "0.adult-2"]]
            },  
            "frame-2": {
                "descriptions": ["The baby drinks from the bottle.", "The adult looks at the baby while feeding."],
                "triplets": [["1.baby-6", "12.drinking from", "15.bottle-4"], ["0.adult-2", "30.looking at", "1.baby-6"]]
            },  
            "frame-3": {
                "descriptions": ["The baby holds onto the bottle while drinking.", "The adult supports the bottle in the baby's hands."],
                "triplets": [["1.baby-6", "20.holding", "15.bottle-4"], ["0.adult-2", "53.toward", "1.baby-6"]]
            },
            "frame-4": {
                "descriptions": ["The adult gently caresses the baby's head.", "The baby continues to drink from the bottle."],
                "triplets": [["0.adult-2", "4.caressing", "1.baby-6"], ["1.baby-6", "12.drinking from", "15.bottle-4"]]
            },
            "frame-5": {
                "descriptions": ["The baby smiles up at the adult.", "The adult smiles back, holding the bottle."],
                "triplets": [["1.baby-6", "30.looking at", "0.adult-2"], ["0.adult-2", "30.looking at", "1.baby-6"], ["0.adult-2", "20.holding", "15.bottle-4"]]
            },
            "frame-6": {
                "descriptions": ["The adult places the baby on the ground.", "The baby crawls next to the adult."],
                "triplets": [["0.adult-2", "47.stepping on", "116.floor-0"], ["1.baby-6", "31.lying on", "116.floor-0"], ["1.baby-6", "0.beside", "0.adult-2"]]
            },
            "frame-7": {
                "descriptions": ["The adult picks up a toy for the baby.", "The baby watches, curious."],
                "triplets": [["0.adult-2", "36.picking", "108.toy-9"], ["1.baby-6", "30.looking at", "0.adult-2"]]
            },
            "frame-8": {
                "descriptions": ["The adult and baby play with the toy together.", "They are both sitting on the floor, laughing."],
                "triplets": [["0.adult-2", "38.playing with", "108.toy-9"], ["1.baby-6", "38.playing with", "108.toy-9"], ["0.adult-2", "45.sitting on", "116.floor-0"], ["1.baby-6", "45.sitting on", "116.floor-0"]]
            },
            "st progression": "The scene portrays a nurturing interaction between an adult and a baby. The adult holds a bottle and feeds the baby, while the baby engages by drinking and eventually holding the bottle. Afterward, they play together with a toy, sharing smiles and laughter."
        }
        #sg_end
        
        
        From the provided video which contains 8 frames, generate triplets.
        Please give the output in the format given above.
"""






Task_description_v10 = f"""The objects_entity lexicon containing 35 lexemes is numbered as follows:""" + vidvrd_objects_numbered + """\n\
    and relations_entity lexicon containing 132 lexemes is numbered as follows:""" +  vidvrd_predicates_numbered + """\n\
    
    Given the objects and relations lexeme, the task is to generate triplets from the provided video in the form of [objects_entity-id lexicon, relations_entity lexicon, objects_entity-id lexicon]. 
    The id is randomly assigned to each object-entity to ensure uniqueness and tracking throughout the video.
    Make sure the index of the objects_entity and relations_entity is maintained.
    Select relations_entity which best describes the relation between two objects.

    Note: It is possible that exact relations_entity or objects_entity might not be visible in the video, but those can be aligned, refer below examples.
    example-1: the relations_entity 'being touched and leaned over by' can be mapped to '41.touch'.
    example-2: the relations_entity 'sitting straighter' can be mapped to '75.sit'.
    example-3: the objects_entity 'rider' can be mapped to '2.person'.
    example-4: the objects_entity 'deer' can be mapped to '1.antelope'.

    In-context Example 1:
        #sg_start
        {
            "scene": { 
                "description" : "Two lions move through the savanna, with the larger lion leading and the smaller lion following behind. Their positions emphasize their spatial relationship and size difference.",
            },
            "f1": {
                "descriptions": ["The larger lion walks in front of the smaller lion.","The smaller lion stands behind the larger lion."],
                "triplets": [["10.lion-0", "18.walk front", "10.lion-1"], ["10.lion-1", "13.stand behind", "10.lion-0"]]
            },  
            "f2": {
                "descriptions": ["The larger lion continues walking ahead.","The smaller lion walks behind, keeping pace."],
                "triplets": [["10.lion-0", "18.walk front", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },  
            "f3": {
                "descriptions": ["The larger lion is taller than the smaller lion.","The smaller lion walks behind, following closely."],
                "triplets": [["10.lion-0", "3.taller", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },
            "f4": {
                "descriptions": ["The larger lion walks right while the smaller lion follows behind.","The size difference between the lions remains visible."],
                "triplets": [["10.lion-0", "14.walk right", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },
            "f5": {
                "descriptions": ["The larger lion walks toward the front, leading the way.","The smaller lion follows behind, staying close."],
                "triplets": [["10.lion-0", "25.walk toward", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },
            "f6": {
                "descriptions": ["The larger lion walks next to the smaller lion for a moment.","The lions maintain their positions."],
                "triplets": [["10.lion-0", "15.walk next to", "10.lion-1"], ["10.lion-1", "10.next to", "10.lion-0"]]
            },
            "f7": {
                "descriptions": ["The larger lion walks past the smaller lion.","The smaller lion stands behind, watching."],
                "triplets": [["10.lion-0", "17.walk past", "10.lion-1"], ["10.lion-1", "13.stand behind", "10.lion-0"]]
            },
            "f8": {
                "descriptions": ["The lions stop moving, the larger one still in front of the smaller lion.","The smaller lion sits behind the larger one."],
                "triplets": [["10.lion-0", "87.stop front", "10.lion-1"], ["10.lion-1", "8.sit behind", "10.lion-0"]]
            },
            "st progression": "The scene shows the movement and spatial relationship of the two lions in the savanna. The larger lion consistently leads in size and position, with the smaller lion following closely behind or beside, eventually stopping behind the larger lion."
        }
        #sg_end

    In-context Example 2:
        #sg_start
        {
            "scene": { 
                "description" : "Two bicycles and two people are interacting as they move through the scene, with the people riding and maneuvering the bicycles. The scene highlights their movements and spatial relationships.",
            },
            "f1": {
                "descriptions": ["The first bicycle moves right of the second bicycle.","The first bicycle moves in front of the second bicycle."],
                "triplets": [["5.bicycle-0", "57.move right", "5.bicycle-1"], ["5.bicycle-0", "45.move front", "5.bicycle-1"]]
            },  
            "f2": {
                "descriptions": ["Both bicycles move together side by side.","The second bicycle follows the first."],
                "triplets": [["5.bicycle-0", "61.move with", "5.bicycle-1"], ["5.bicycle-1", "61.move with", "5.bicycle-0"]]
            },  
            "f3": {
                "descriptions": ["The second bicycle moves behind the first.","The second bicycle shifts to the left of the first."],
                "triplets": [["5.bicycle-1", "19.move behind", "5.bicycle-0"], ["5.bicycle-1", "58.move left", "5.bicycle-0"]]
            },
            "f4": {
                "descriptions": ["The first bicycle moves beneath the person.","The person sits above the first bicycle, riding it."],
                "triplets": [["5.bicycle-0", "46.move beneath", "2.person-3"], ["2.person-3", "32.sit above", "5.bicycle-0"]]
            },
            "f5": {
                "descriptions": ["The first bicycle moves right of another person.","The first bicycle moves in front of the other person."],
                "triplets": [["5.bicycle-0", "57.move right", "2.person-5"], ["5.bicycle-0", "45.move front", "2.person-5"]]
            },
            "f6": {
                "descriptions": ["The other person stands behind the first bicycle.","The other person stands left of the first bicycle."],
                "triplets": [["2.person-5", "37.behind", "5.bicycle-0"], ["2.person-5", "44.left", "5.bicycle-0"]]
            },
            "f7": {
                "descriptions": ["The second bicycle moves behind the person.","The second bicycle moves left of the person."],
                "triplets": [["5.bicycle-1", "19.move behind", "2.person-3"], ["5.bicycle-1", "58.move left", "2.person-3"]]
            },
            "f8": {
                "descriptions": ["The person sits above the second bicycle and rides it.","The two people align with each other, moving through the scene."],
                "triplets": [["2.person-5", "32.sit above", "5.bicycle-1"], ["2.person-5", "61.move with", "2.person-3"]]
            },
            "st progression": "The two bicycles and two people move through the scene, interacting closely. The bicycles are ridden by the people, with movements emphasizing spatial relationships such as moving left, right, in front, and behind."
        }
        #sg_end
        
        From the provided video which contains 8 frames, generate triplets:
        Note: Triplets should describe spatial(objects placement in the scene with referece to other objects) and action relationships between objects present in the video. Please give the output in the format given above.
"""
#Note: Triplets should describe whats happening in the video. Please give the output in the format given above.


Task_description_v9 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Subjects/Objects: Entities involved in the scene.
        - Relation:
            - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
            - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
            - Temporal Changes: How the relationships or positions change across different frames.

    Other important things to consider:
        - You are provided with specific lists of Relations/Actions/Temporal Changes,and you must only consider those when generating scene graph triplets.
        - Do not use any relations if it does not accurately describe an action or spatial relationship between the objects present in the video.
    
    In-context Example 1:
        Given list of Relations=[jump front,sitting on,walk behind,standing next to,stand front,stand behind,walk front,reaching for,cathcing,holding,chasing,walking toward]

        #sg_start
        {
            "scene": {
                "description": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door."
            },
            "f1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "chair-9"], ["table-5", "next to", "person-7"]]
            },  
            "f2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "book-4"]]
            },  
            "f3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "book-4"]]
            },
            "f4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "door-0"]]
            },
            "st progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:
        Given list of Relations=[jump front,running in,jumping over,walk behind,in front of,stand front,stand behind,walk front,reaching for,cathcing,holding,chasing,walking toward]

        #sg_start
        {
            "scene": { 
                "description" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            },
            "f1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "f2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "f3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "f4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "st progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end

    Note that only given list of relations are used to describe video and generate triplets
    Now given list of Relations=[{List_of_predicates}]
"""



Task_description_v8 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Subjects/Objects: Entities involved in the scene.
        - Relation:
            - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
            - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
            - Temporal Changes: How the relationships or positions change across different frames.

    Other important things to consider:
        - You are provided with specific lists of Relations/Actions/Temporal Changes,and you must only consider those when generating scene graph triplets.
        - Do not use any Relations/Actions/Temporal Changes if it does not describe an action or spatial relationship between the objects present in the video.
        - If Subjects/Objects from the provided list does not appear in the video, do not generate triplet for using such Subjects/Objects.
    
    In-context Example 1:
        Given list of Objects=[mountain, river,person, book, door,horse, train, plane,chair,monkey] and 
        Relations/Actions/Temporal Changes=[jump front,sitting on,walk behind,standing next to,stand front,stand behind,walk front,reaching for,cathcing,holding,chasing,walking toward]

        #sg_start
        {
            "scene": {
                "description": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door."
            },
            "f1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "chair-9"], ["table-5", "next to", "person-7"]]
            },  
            "f2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "book-4"]]
            },  
            "f3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "book-4"]]
            },
            "f4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "door-0"]]
            },
            "st progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:
        Given list of Objects=[fence,mountain,river,person, book,ball,door,horse,park,train, plane,chair,monkey, dog] and 
        Relations/Actions/Temporal Changes=[jump front,running in,jumping over,walk behind,in front of,stand front,stand behind,walk front,reaching for,cathcing,holding,chasing,walking toward]

        #sg_start
        {
            "scene": { 
                "description" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            },
            "f1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "f2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "f3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "f4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "st progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end

    Note that only given list of relations are used to describe video and generate triplets
    Now given list of Objects=[{List_of_objects}] and 
    Relations=[{List_of_predicates}]
"""

Task_description_v7 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Subjects/Objects: Entities involved in the scene.
        - Relation:
            - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
            - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
            - Temporal Changes: How the relationships or positions change across different frames.

    Other important things to consider:
        - You are provided with specific lists of Relations/Actions/Temporal Changes,and you must only consider those when generating scene graph triplets.
        - Do not use any relations if it does not accurately describe an action or spatial relationship between the objects present in the video.
    
    In-context Example 1:
        Given list of Relations=[jump front,sitting on,walk behind,standing next to,stand front,stand behind,walk front,reaching for,cathcing,holding,chasing,walking toward]

        #sg_start
        {
            "scene": {
                "description": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door."
            },
            "f1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "chair-9"], ["table-5", "next to", "person-7"]]
            },  
            "f2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "book-4"]]
            },  
            "f3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "book-4"]]
            },
            "f4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "door-0"]]
            },
            "st progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:
        Given list of Relations=[jump front,running in,jumping over,walk behind,in front of,stand front,stand behind,walk front,reaching for,cathcing,holding,chasing,walking toward]

        #sg_start
        {
            "scene": { 
                "description" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            },
            "f1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "f2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "f3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "f4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "st progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end

    Note that only given list of relations are used to describe video and generate triplets
    Now given list of Relations=[{List_of_predicates}]
"""


# For example: 
# if subjects/objects: [person, cup, child, sofa, chair, table, cake] and predicates: [running on, sitting on, holding, picking, jumping] then based on the scene triplets can be #sg_start [person-8,holding,cup-2];[child-3,running on,floor-0];[child-6,sitting on,sofa-10] #sg_end  

Task_description_v6 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Objects: Entities involved in the scene.
        - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
        - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
        - Temporal Changes: How the relationships or positions change across different frames.
    
    In-context Example 1:
        #sg_start
        {
            "Scene": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door.",
            "F1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "Chair-9"], ["Table-5", "next to", "person-7"]]
            },  
            "F2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "Book-4"]]
            },  
            "F3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "Book-4"]]
            },
            "F4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "Door-0"]]
            },
            "ST Progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:

        #sg_start
        {
            "Scene" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            "F1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "Park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "F2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "F3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "F4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "ST Progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end
"""



Task_description_v5 = """
Task: Describe provided video in detailed manner in the triplets format [subject-#id, predicate, object-#id] using provided list of subjects/objects and predicates.

- subject is an entity who is performing action in the scene, or entity selected to describe it's spatial placement in the scene (e.g. antelope, table, cat, person etc.)
- The #id in (subject-#id and object-#id) is a randomly assigned unique identifier used to distinguish objects of the same category and track the objects throughout the scene.
- predicate describes what subject is doing in the scene with object or describes subject's placement in the scene with respect to other objects. (e.g. holding, standing next to, drinking etc.)
- object is an entity which receives the action done by subject or its used as a reference to which subjects spatial position is given.


In-context Example 1:
    #sg_start
    {
        "Scene": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door.",
        "F1": {
            "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
            "triplets": [["person-7", "sitting on", "Chair-9"], ["Table-5", "next to", "person-7"]]
        },  
        "F2": {
            "descriptions": ["The person is reaching out to pick up the book from the table."],
            "triplets": [["person-7", "reaching for", "Book-4"]]
        },  
        "F3": {
            "descriptions": ["The person picks up the book and is now holding it."],
            "triplets": [["person-7", "holding", "Book-4"]]
        },
        "F4": {
            "descriptions": ["The person walks away from the chair toward the door."],
            "triplets": [["person-7", "walking toward", "Door-0"]]
        },
        "ST Progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
    }
    #sg_end


In-context Example 2:
    #sg_start
    {
        "Scene" : "A dog runs in a park, jumps over a fence, and chases a ball.",
        "F1": {
            "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
            "triplets": [["dog-2", "running in", "Park-4"], ["fence-2", "in front of", "dog-2"]]
        },  
        "F2": {
            "descriptions": ["The dog jumps over the fence."],
            "triplets": [["dog-2", "jumping over", "fence-2"]]
        },  
        "F3": {
            "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
            "triplets": [["dog-2", "chasing", "ball-3"]]
        },
        "F4": {
            "descriptions": ["The dog catches up to the ball."],
            "triplets": [["dog-2", "reaching", "ball-3"]]
        },
        "ST Progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
    }
    #sg_end

Now, Describe provided video in detailed manner, Use the provided subjects/objects: {List_of_objects} and predicates: {List_of_predicates}
"""


# Task_description = """
# Task: Describe provided video in detailed manner using Scene Graph Triplets in the format [subject-#id, predicate, object-#id].

# - subject is an entity who is performing action in the scene, or entity selected to describe it's spatial placement in the scene (e.g. antelope, table, cat, person etc.)
# - The #id in (subject-#id and object-#id) is a unique identifier used to distinguish objects of the same category (e.g., adult-1, adult-3, table-4 etc.)
# - predicate describes what subject is doing in the scene with object or describes subject's placement in the scene with respect to other objects. (e.g. holding, standing next to, drinking etc.)
# - object is an entity which receives the action done by subject or its used as a reference to which subjects spatial position is given.

# For example: [person-1,holding,cup-2];[child-3,running on,floor-0];[child-3,sitting on,sofa-10]

# Now follow the steps below.
# 1. Generate triplets from the provided video in the format [subject-#id, predicate, object-#id], use the provided subjects/objects and predicate lists to generate triplets: subjects/objects: {List_of_objects} and predicates: {List_of_predicates}.
# """


# Task_description = """
#     Task: Describe video using Scene Graph Triplets in the format [subject-#id, predicate, object-#id].

#     Definitions:
#     - subject-#id: A visual object or noun in the video that performs an action or plays a role(actor) in the scene. The #id is a unique identifier used to distinguish objects of the same category (e.g., multiple people).
#     - predicate: Describes the action (verb) or spatial relationship between objects in the scene.
#     - object-#id: A visual object or noun in the video affected by the subject's action or shows spatial relationship with respect to other objects in the scene.

#     Example 1:
#         - [person-1,holding,cup-2];[child-3,running on,floor-0]
    
#     Notes: 
#         - Remember to assign unique #id to all objects without repeating the same objects in the scene.
#         - There can be N number of same category of objects but make sure they are visible in the video before assigning them the #id.

#     Now follow the steps below:

#     1. From the given video, construct meaningful scene graph triplets in the format [subject-#id, predicate, object-#id].
#         - Assign a unique #id to each object if more than one of the same category is present.
#         - Ensure that the subjects/objects in the triplets are visible in the video.
#     """

# 2. Using the output from Step 1, map each subject, object, and predicate to the corresponding entries in the provided list of subjects/objects {List_of_objects} and predicates {List_of_predicates}.
#         - Keep the #ids consistent during mapping.
#         - Ensure that the meaning of the original triplets remains unchanged during the mapping process.

# Notes:
#     - Triplets must be unique and describe meaningful interactions or spatial relations between objects in the scene.
#     - Use common sense when generating or mapping the triplets.
#     - If an object in the provided list is not visible in the video, do not use it to generate triplets.

Task_description_v3 = """
    Task: Scene graph triplet prediction in the format [subject-#id, predicate, object-#id] from the given video.
        - subject is a visual object or noun which is present in the video which can be seen as an entity who is performing action or is an actor in the scene.
            - here #id in subject-#id describes or tags the object visible in the scene to have uniqueness, if more than one objects of similar category are visible in the scene.
        - predicate can be an action(verb) which describes what the subject is doing in the scene or spatial relation for how objects are positioned to each other in the scene.
        - object is a visual object or noun which is present in the video which is described by the predicate and subject combination, this object is affected by subject and predicate.

    Notes: 
        - Remember to assign unique #id to all objects without repeating the same objects in the scene.
        - There can be N number of same category of objects but make sure they are visible in the video before assigning them the #id.

    
    Now follow the below steps,

    1. Describe the given video in the form of meaningful scene graph triplets in the format [subject-#id, predicate, object-#id]
    2. Take each subjects/objects from the step-1 and use the following list of subject/objects {List_of_objects} and predicates {List_of_predicates} and align subjects/objects without changing the #id, or predicates such that it does not change the meaning of the original triplets obtained in step-1.
   
    Notes:
        - Make sure the triplets are meaningful and with common sense and unique
        - Triplets must have subjects/objects and predicates from the provided list above and should be present in the video.
        - Make sure to cover all subject/objects to describe their spatial postions.
        - if objects don't appear in the video which is provided in the list, ignore it.
    """



    # 1. {Detect_Count_objects[v0]}
    # 2. Use provided list of objects {List_of_objects}, and if any of the detected objects in step-1 are semantically same, assign them new category from the provided list but keep the assigned #id.
    # 3. Create scene graph triplets in the format [subject-#id, predicate, object-#id].

#3. From the below provided list of objects check if objects obtained in step-2 are semantically same in the below provided list, if so use the object name from the provided list of objects

# 4. Now based on the provided video identify the relationships between objects and use objects constructed in step 3 and provided list of predicates to construct meaningful scene graph triplets the format [subject-#id, predicate, object-#id]
#         - Make sure the triplets are meaningful and unique
#         - Use objects name and #id from step-3



Task_description_v2 = """
    Your task is to generate meaningful scene graph triplets for the provided video, to do so follow the steps below.

    1. Identify objects present in the video.
    2. Assign uniqe #id to all the objects obtained in step-1 for their identity and tracking (e.g dog-1, dog-2, cat-9, mouse-8), the #id assignment should stay consistant throughout the video. The #ids count should match with total number of objects in video.
    3. Use the provided list of objects and replace if the objects are semantically same.
        - For example, if video consist of a person, and provided list has adult as an object elment then use adult as an object.
    4. For each objects from step-3 and provided predicates below, construct scene graph triplets in the format [subject-#id, predicate, object-#id] which describes the provided video.
        - Make sure the triplets are meaningful and unique
    """
Task_description_v1 = """
    Your task is to generate meaningful scene graph triplets for the provided video, to do so follow the steps below.

    1. Identify objects present in the video.
        1.1 Use below provided list of objects and use objects name from the list if they are semantically the same.
    2. Assign uniqe #id to each objects obtained in step-1 for their identity and tracking (e.g dog-1, dog-2, cat-9, mouse-8), the #id assignment should stay consistant throughout the video. The max #ids count should match with total number of objects in step-1.
    3. Using objects obtained in step-3, contruct meaningful scene graph triplets the format [subject-#id, predicate, object-#id] from the video.
        - Note: Use below provided list of predicates to construct the triplets.
    """

Task_description_v0 = """
    Your task is to generate meaningful scene graph triplets for the provided video, to do so follow the steps below.

    1. Use the provided list to identify objects in the video. Only consider objects that appear both in the video and are on the list.
    2. Use the provided list of spatial predicates and action predicates that are happening in the video between objects.
    3. Finally, construct meaningful scene graph triplets in the format [subject-#id, predicate, object-#id] with a common sense, where the subject is performing the action (predicate), and the object is receiving the action. The #id refers to the unique identifier assigned to each object in the video.
    
    Important things to consider: 
        1. Ignore any objects in the video that are not on the list, as well as any objects on the list that do not appear in the video
        2. Track the objects in the consecutive frames to reassign the same #ids to the objects.
        3. Make sure the triplets constructed are logical and with common sense.
    """