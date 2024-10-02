
List_of_objects = []
List_of_predicates = [] 

# Example-1: [cat-1, chasing, mouse-2]
#     - cat is a subject and #id 1 is assigned to cat(subject).
#     - chasing is a predicate/action which cat-1 is performing
#     - mouse-2 is an object and #id 2 is assigned mouse(object) which is affected by cat-1(subject) and chasing(predicate)
# Example-2 [cat-4, standing next, cat-2]
#     - cat-4 is a subject and #id 4 is assigned to cat(subject).
#     - standing next is a sptial predicate which describes cat-4(subject) in the scene
#     - cat-2 is an object and #id 2 is assigned cat(object) which is affected by cat-4(subject) and standing next(predicate)

# Task: Generate scene graph triplets from the given video in the format [subject-#id, predicate, object-#id].




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
                "triplets": [["Person-7", "sitting on", "Chair-9"], ["Table-5", "next to", "Person-7"]]
            },  
            "F2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["Person-7", "reaching for", "Book-4"]]
            },  
            "F3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["Person-7", "holding", "Book-4"]]
            },
            "F4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["Person-7", "walking toward", "Door-0"]]
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
                "triplets": [["Dog-2", "running in", "Park-4"], ["Fence-2", "in front of", "Dog-2"]]
            },  
            "F2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["Dog-2", "jumping over", "Fence-2"]]
            },  
            "F3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["Dog-2", "chasing", "Ball-3"]]
            },
            "F4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["Dog-2", "reaching", "Ball-3"]]
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
            "triplets": [["Person-7", "sitting on", "Chair-9"], ["Table-5", "next to", "Person-7"]]
        },  
        "F2": {
            "descriptions": ["The person is reaching out to pick up the book from the table."],
            "triplets": [["Person-7", "reaching for", "Book-4"]]
        },  
        "F3": {
            "descriptions": ["The person picks up the book and is now holding it."],
            "triplets": [["Person-7", "holding", "Book-4"]]
        },
        "F4": {
            "descriptions": ["The person walks away from the chair toward the door."],
            "triplets": [["Person-7", "walking toward", "Door-0"]]
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
            "triplets": [["Dog-2", "running in", "Park-4"], ["Fence-2", "in front of", "Dog-2"]]
        },  
        "F2": {
            "descriptions": ["The dog jumps over the fence."],
            "triplets": [["Dog-2", "jumping over", "Fence-2"]]
        },  
        "F3": {
            "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
            "triplets": [["Dog-2", "chasing", "Ball-3"]]
        },
        "F4": {
            "descriptions": ["The dog catches up to the ball."],
            "triplets": [["Dog-2", "reaching", "Ball-3"]]
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