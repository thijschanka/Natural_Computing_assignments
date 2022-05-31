from Backtracking.Constraints import Constraints
import numpy as np

class PuzzleState:
    EMPTY = False
    BLOCK = True

    def __init__(self, constraints):
        self.constraints = constraints
        self._state = [
            [None for _ in range(constraints.width)] for _ in range(constraints.height)
        ]

    def set(self, row, column, value):
        self._state[row][column] = value

    def set_row(self, row, values):
        self._state[row] = values

    def get(self, row, column):
        return self._state[row][column]
    
    def get_constraints(self):
        return self.constraints
    
    def __str__(self):
        return "\n".join(
            [ "┌" + "".join('─' for _ in range(self.constraints.width)) + "┐" ] + \
            [ "│" + "".join(
                        "█" if self._state[i][j] else " " for j in range(self.constraints.width) 
                    ) + "│"
                    for i in range(self.constraints.height) ] + \
            [ "└" + "".join('─' for _ in range(self.constraints.width)) + "┘"]
        )
        
    def validate(self, completed_rows):
        if completed_rows <= 0:
            return True

        completed_rows += 1
        
        for i in range(self.constraints.width):
            column_constraints = self.constraints.columns[i]

            # if there are no blocks in the current column
            if len(column_constraints) == 0:
                # return false if there is any block
                for j in range(completed_rows):
                    if self.get(j, i):
                        return False
                
                # column is valid
                continue

            in_block = False # flag if the current position is in a block
            block_index = 0 # the index of the next block
            num_cells = None # the number of remaing cells in the current block
            
            for j in range(completed_rows):
                if self.get(j, i): # the current cell is occupied
                    if in_block:
                        num_cells -= 1  # consume one cell of the remaining ones
                        if num_cells < 0:
                            # there are more cells in the block than in the constraint
                            return False
                    else:
                        if block_index >= len(column_constraints):
                            return False # a new block starts but there are no more in the constraints

                        num_cells = column_constraints[block_index] - 1
                        block_index += 1
                        in_block = True
                elif in_block:
                    if num_cells != 0: 
                        return False # if not all cells were consumed the state is not valid
                    in_block = False
            
            if completed_rows == self.constraints.height and block_index != len(column_constraints):
                return False # there were too few blocks in the current state
            
            # check if the column can't be completed with the remaining blocks
            remaining_cells = self.constraints.height - completed_rows
            remaining_constraints = column_constraints[block_index:]
            if sum(remaining_constraints) + len(remaining_constraints) - 1 > remaining_cells:
                return False

        return True # no errors were found so the state is valid



    def constraint_violations(self) -> int:
        state = np.array(self._state)

        for i, col_constraint in enumerate(self.constraints.columns):
            pass

        raise NotImplementedError