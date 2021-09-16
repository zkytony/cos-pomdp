
def _test_path_result_integrity():n
    path_result1 =  {'scene': 'FloorPlan22',
                     'target': 'Knife',
                     'shortest_path': [{'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                       {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                       {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                       {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                       {'x': -1.25, 'y': 0.9009991884231567, 'z': 0.25},
                                       {'x': -1.0, 'y': 0.9009991884231567, 'z': 0.5},
                                       {'x': -0.75, 'y': 0.9009991884231567, 'z': 0.75},
                                       {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0},
                                       {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}],
                     'shortest_path_distance': 1.4142135623730951,
                     'actual_path': [{'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0},
                                     {'x': -1.25, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': -1.25, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': -1.25, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': -1.25, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': -1.25, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': -1.0, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -1.0, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.75, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.5, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.5, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.5, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.25, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.25, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.25, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.25, 'y': 0.9009991884231567, 'z': -0.5},
                                     {'x': -0.0, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': -0.0, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': 0.25, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': 0.5, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': 0.5, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': 0.5, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': 0.5, 'y': 0.9009991884231567, 'z': -0.25},
                                     {'x': 0.75, 'y': 0.9009991884231567, 'z': -0.0},
                                     {'x': 0.75, 'y': 0.9009991884231567, 'z': -0.0},
                                     {'x': 0.75, 'y': 0.9009991884231567, 'z': -0.0},
                                     {'x': 0.75, 'y': 0.9009991884231567, 'z': -0.0},
                                     {'x': 0.75, 'y': 0.9009991884231567, 'z': -0.0},
                                     {'x': 0.75, 'y': 0.9009991884231567, 'z': -0.0},
                                     {'x': 0.75, 'y': 0.9009991884231567, 'z': -0.0},
                                     {'x': 0.5, 'y': 0.9009991884231567, 'z': 0.25},
                                     {'x': 0.25, 'y': 0.9009991884231567, 'z': 0.5},
                                     {'x': 0.25, 'y': 0.9009991884231567, 'z': 0.5},
                                     {'x': 0.25, 'y': 0.9009991884231567, 'z': 0.5},
                                     {'x': 0.0, 'y': 0.9009991884231567, 'z': 0.75},
                                     {'x': 0.0, 'y': 0.9009991884231567, 'z': 0.75},
                                     {'x': 0.0, 'y': 0.9009991884231567, 'z': 0.75}],
                     'actual_path_distance': 3.724873734152917, 'success': True}


    {'scene': 'FloorPlan22', 'target': 'Knife', 'shortest_path': [{'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.25, 'y': 0.9009991884231567, 'z': 0.25}, {'x': -1.0, 'y': 0.9009991884231567, 'z': 0.5}, {'x': -0.75, 'y': 0.9009991884231567, 'z': 0.75}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}], 'shortest_path_distance': 1.4142135623730951, 'actual_path': [{'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.25, 'y': 0.9009991884231567, 'z': 0.25}, {'x': -1.0, 'y': 0.9009991884231567, 'z': 0.5}, {'x': -0.75, 'y': 0.9009991884231567, 'z': 0.75}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}], 'actual_path_distance': 1.4142135623730951, 'success': True}{'scene': 'FloorPlan22', 'target': 'Knife', 'shortest_path': [{'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.25, 'y': 0.9009991884231567, 'z': 0.25}, {'x': -1.0, 'y': 0.9009991884231567, 'z': 0.5}, {'x': -0.75, 'y': 0.9009991884231567, 'z': 0.75}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}], 'shortest_path_distance': 1.4142135623730951, 'actual_path': [{'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.5, 'y': 0.9009991884231567, 'z': 0.0}, {'x': -1.25, 'y': 0.9009991884231567, 'z': 0.25}, {'x': -1.0, 'y': 0.9009991884231567, 'z': 0.5}, {'x': -0.75, 'y': 0.9009991884231567, 'z': 0.75}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}, {'x': -0.5, 'y': 0.9009991884231567, 'z': 1.0}], 'actual_path_distance': 1.4142135623730951, 'success': True}
