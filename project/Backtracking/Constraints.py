class Constraints:
    def __init__(self, width, height, rows, columns):
        self.width = width
        self.height = height
        self.rows = rows
        self.columns = columns

    def get_constraints(json_object):
        return [], Constraints(json_object['width'],
                               json_object['height'],
                               json_object['rows'],
                               json_object['columns'])