InContext_Examples = """
    Example 1:
    Unique Objects in the video: [cat, dog, mountain]
    Unique Predicates: [running chasing, behind]
    Unique Scene graph triplets in the video: [dog-1, chasing, cat-2];[dog-1,behind, cat-2];[dog-0,chasing,cat-2];[dog-0,behind, cat-2];
    Note: In Example 1, there are two dogs present in the scene, which is why dog-0 and dog-1 are assigned. Since the mountain does not appear in the video, no triplet is provided for it.

    Example 2:
    Unique Objects in the video: [child, adult, cake, candle, chair, table, floor]
    Unique Predicates: [sitting on, standing on, on]
    Unique Scene graph triplets in the video: [child-1,sitting on, chair-7];[adult-5,standing on,floor-0];[cake-5,on,table-9];[candle-8,on,cake-5]

    Example 3:
    Unique Objects in the video: [deer]
    Unique Predicates: stand left, stand right, stand front, stand behind
    Unique Scene graph triplets: [deer-1,stand left,deer-2];[deer-1,stand front,deer-3];[deer-2,stand right,deer-3];[deer-3,stand behind,deer-1]
    """