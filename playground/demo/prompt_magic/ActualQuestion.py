from prompt_magic import ListOfObjectsAndPredicates
import importlib
importlib.reload(ListOfObjectsAndPredicates)

# Actual_instructions = """
#     For the step 3 use the list of Objects: [cattle, house, door, mountains, bottle, table, sofa]
#     For the step 4 use the list of Predicates: [stand behind, stand front,stand next to,walk next to,walk front,walk behind,walk with]
#     """

Actual_instructions = f"""
    Use the list of Objects: {ListOfObjectsAndPredicates.ListOfObjectsAndPredicates['cowseatinggrass']['objects']}
    Use the list of Predicates: {ListOfObjectsAndPredicates.ListOfObjectsAndPredicates['cowseatinggrass']['predicates']}
    """ 

Actual_instructions_v0 = """
    Now from the given video generate scene graph
    Unique Objects in the video: [cattle, house, door, mountains, bottle, table, sofa]
    Unique Predicates: [stand behind, stand front,stand next to,walk next to,walk front]
    Unique Scene graph triplets in the video:
    """