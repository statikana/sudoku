import numpy as np
from itertools import product  # itertools.product my beloved
from puzzles import *

class Sudoku:
    def __init__(self, board: list[list[int]] | np.ndarray):
        self.board: np.ndarray = np.array(board)
        self.box_height: int = 3
        self.box_width: int = 3

    def is_solved(self):
        return all(len(row != 0) == len(row) for row in self)

    def get_box_values(self, n: int):
        box_row: int = n // self.height
        box_col: int = n % self.width
        return self.board[
            box_row * self.box_height : box_row * self.box_height + self.box_height,
            box_col * self.box_width : box_col * self.box_width + self.box_width,
        ]

    def box_group(self, row: int, col: int):
        box_row: int = row // (self.height // self.box_height)
        box_col: int = col // (self.width // self.box_width)

        return set(
            product(
                range(
                    box_row * self.box_height,
                    box_row * self.box_height + self.box_height,
                ),
                range(
                    box_col * self.box_width, box_col * self.box_width + self.box_width
                ),
            )
        )

    def interacting_indicies(self, row: int, col: int):
        """
        Returns a list of all possible indicies of the board which may conflict with the given index (all other
        indices in the row, column, and box)
        """
        # (4, 5)
        #, (4, x)
        # (4, 0)
        # (4, 1)
        #, (4, ...) -> (4, 8)
        rows = {(row, col_t) for col_t in range(0, self.width)}
        
        # (4, 5)
        # (0, 5)
        # (1, 5)
        # (... 5) -> (8, 5)
        cols = {(row_t, col) for row_t in range(0, self.height)}
        
        box = self.box_group(row, col)

        return (rows | cols | box) - {(row, col)}

    def solve(self):
        # fmt: off
        
        has_changed: bool = True  # keeps track if the last iteration over possible values resulted in any changes to the board
        
        while has_changed: 
            has_changed = False
            # `possible` reprensts all obviously possible
            # values for each square (eg. [0, 1, 1, 0, 0, 0, 0, 0, 0] 
            # means that only numbers "2" and "3" are possible at this spot due
            # immedietly available conflicts (row, col, box)
            possible_key = self.create_possible()
            
            # find any spots in possible where only possible option
            
            for row, col in product(range(self.height), range(self.width)):
                
                # get current known possible values
                possible = possible_key[row, col]
                
                if sum(possible) == 1: # only one possible value (True=1, False=0)
                    value = np.where(possible == True)[0][0] + 1  # index + 1
                    possible_key[row, col] = np.zeros((9,))
                    self.board[row, col] = value
                    has_changed = True
                
            print(self)        
        print("done with immedietly apparent fills")
        print("[n possible digits: n squares with respective count]")
        final_possible = self.create_possible()
        count = {}
        for row, col in product(range(self.height), range(self.width)):
            possible = final_possible[row, col]
            n_possible = int(sum(possible))
            if n_possible in count:
                count[n_possible] += 1
            else:
                count[n_possible] = 1
        
        for n, b in count.items():
            print(f"{str(n).rjust(6) if n != 0 else 'filled'}: {b}")
    
            
    def create_possible(self):
        # fmt: off
        possible = np.ones((self.height, self.width, 9))
        given = self.board != 0
        possible[given] = np.zeros((9,))  # any existing spots are set to all false possibles
        
        for row, col in product(range(self.height), range(self.width)):
            if given[row, col]:
                value = self.board[row, col]
                inter = self.interacting_indicies(row, col)
                for i, j in inter:
                    possible[i, j, value-1] = False
        
        return possible
        # fmt: on
    

    @property
    def height(self):
        return len(self.board)

    @property
    def width(self):
        return len(self.board[0])

    @classmethod
    def random(cls, height: int = 9, width: int = 9) -> "Sudoku":
        board = np.random.randint(1, 10, (height, width))
        return cls(board)

    def __iter__(self):
        return iter(self.board)

    def __getitem__(self, i: int):
        return self.board.__getitem__(i)
    
    def __repr__(self):
        string = ""
        top = "┌─────────┬─────────┬─────────┐"
        mid = "├─────────┼─────────┼─────────┤"
        bot = "└─────────┴─────────┴─────────┘"
        
        string += top + "\n"
        for i, row in enumerate(self.board):
            string += "│"
            for j, val in enumerate(row):
                string += f" {val if val != 0 else ' '} "
                if j % 3 == 2 and j != 8:
                    string += "│"
            string += "│\n"
            if i % 3 == 2 and i != 8:
                string += mid + "\n"
        string += bot
        return string

    
                    
suds = Sudoku(medium_1)
print(suds.interacting_indicies(4, 5))
print(suds)
suds.solve()
print(suds)

# find 2 boxes with matching pairs of possible values
# that are intersecting
# subtract from other intersecting boxes in the same intersection context (row, col, box)

# then, extend to 3 with matching triplets of possible values
# ...