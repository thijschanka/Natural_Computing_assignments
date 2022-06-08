from itertools import zip_longest

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

    def get_state(self):
        return self._state

    def get_constraints(self):
        return self.constraints

    def __str__(self):
        return "\n".join(
            ["┌" + "".join('─' for _ in range(self.constraints.width)) + "┐"] + \
            ["│" + "".join(
                "█" if self._state[i][j] else " " for j in range(self.constraints.width)
            ) + "│"
             for i in range(self.constraints.height)] + \
            ["└" + "".join('─' for _ in range(self.constraints.width)) + "┘"]
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

            in_block = False  # flag if the current position is in a block
            block_index = 0  # the index of the next block
            num_cells = None  # the number of remaing cells in the current block

            for j in range(completed_rows):
                if self.get(j, i):  # the current cell is occupied
                    if in_block:
                        num_cells -= 1  # consume one cell of the remaining ones
                        if num_cells < 0:
                            # there are more cells in the block than in the constraint
                            return False
                    else:
                        if block_index >= len(column_constraints):
                            return False  # a new block starts but there are no more in the constraints

                        num_cells = column_constraints[block_index] - 1
                        block_index += 1
                        in_block = True
                elif in_block:
                    if num_cells != 0:
                        return False  # if not all cells were consumed the state is not valid
                    in_block = False

            if completed_rows == self.constraints.height and block_index != len(column_constraints):
                return False  # there were too few blocks in the current state

            # check if the column can't be completed with the remaining blocks
            remaining_cells = self.constraints.height - completed_rows
            remaining_constraints = column_constraints[block_index:]
            if sum(remaining_constraints) + len(remaining_constraints) - 1 > remaining_cells:
                return False

        return True  # no errors were found so the state is valid

    """
    Error function based on paper: Aloglah, Roba. (2016). An Efficient Genetic Algorithm and Logical Rule for Solving 
    Nonogram Puzzle. international journal of computer science and information security. 14. 
    https://www.academia.edu/30682190/An_Efficient_Genetic_Algorithm_and_Logical_Rule_for_Solving_Nonogram_Puzzle
    https://www.researchgate.net/publication/308784478_An_Efficient_Genetic_Algorithm_and_Logical_Rule_for_Solving_Nonogram_Puzzle
    """
    def error_1(self) -> int:
        state = np.array(self._state)

        height, width = state.shape

        f1 = np.abs([sum(state[i]) - sum(self.constraints.rows[i]) for i in range(height)]).sum()
        f2 = np.abs([sum(state[:, i]) - sum(self.constraints.columns[i]) for i in range(width)]).sum()
        f3 = np.abs([(width - sum(state[i])) - (width - sum(self.constraints.columns[i])) for i in range(height)]).sum()
        f4 = np.abs(
            [(height - sum(state[:, i])) - (height - sum(self.constraints.columns[i])) for i in range(width)]).sum()

        return f1 + f2


    """
    Error function based this repo: 
    https://github.com/morinim/vita/wiki/nonogram_tutorial
    """
    def error_2(self) -> float:
        state = np.array(self._state)

        height, width = state.shape

        current_rows = []
        for i in range(height):
            row = []
            block = 0
            for j in range(width):
                if state[i, j] == 1:
                    block += 1
                else:
                    if block != 0:
                        row.append(block)
                        block = 0
            if block != 0:
                row.append(block)

            current_rows.append(row)

        current_cols = []
        for j in range(width):
            col = []
            block = 0
            for i in range(height):
                if state[i, j] == 1:
                    block += 1
                else:
                    if block != 0:
                        col.append(block)
                        block = 0
            if block != 0:
                col.append(block)

            current_cols.append(col)

        # Calculate differences
        row_error = 0
        for i in range(height):
            cur = current_rows[i]
            const = self.constraints.rows[i]

            for cu, co in zip_longest(cur, const, fillvalue=0):
                row_error += np.abs(cu-co)

        # Calculate differences
        col_error = 0
        for i in range(width):
            cur = current_cols[i]
            const = self.constraints.columns[i]

            for cu, co in zip_longest(cur, const, fillvalue=0):
                col_error += np.abs(cu-co)

        return float(row_error + col_error)
