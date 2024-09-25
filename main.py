from enum import Enum
from typing import Generator
import numpy as np
import numpy.typing as npt
from itertools import product  # itertools.product my beloved
from puzzles import *

class Sudoku:
    def __init__(self, board: list[list[int]] | npt.NDArray[np.int8]) -> None:
        self.board: np.ndarray = np.array(board)
        self.box_height: int = 3
        self.box_width: int = 3

    def is_solved(self) -> bool:
        """
        Whether or not all spots in the board have an answer placed
        """
        return all(all(row) for row in self.board)

    def get_box_values(self, n: int) -> npt.NDArray[np.int8]:
        """
        Gets all values in the given box number [0-8]
        """
        box_row: int = n // self.height
        box_col: int = n % self.width
        return self.board[
            box_row * self.box_height : box_row * self.box_height + self.box_height,
            box_col * self.box_width : box_col * self.box_width + self.box_width,
        ]

    def box_group(self, row: int, col: int) -> set[tuple[int, int]]:
        """
        Gets all values in the same box as the given row and column
        """
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
    
    def box_n(self, row: int, col: int) -> int:
        """
        Gets the box number of the given row and column
        """
        box_row: int = row // (self.height // self.box_height)
        box_col: int = col // (self.width // self.box_width)
        
        return box_row * self.box_height + box_col


    def get_interacting_indicies(self, row: int, col: int):
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
        # replace all singles
        possible = self.create_possible()
        self.replace_singles(possible)
        
        possible = self.resolve_pairs(self.create_possible())
        self.replace_singles(possible)
    
    def replace_singles(self, possible: np.ndarray) -> None:
        """
        Replaces all apparent values in .board using the possibility map given. Only does one pass
        """
        for row, col in product(range(self.height), range(self.width)):
            pk = possible[row, col]
            if sum(pk == 1):
                value = np.where(pk == 1)[0][0] + 1  # index + 1
                possible[row, col] = np.zeros((9,))
                self.board[row, col] = value
            
        
    def least_unresolved(self) -> int:
        """
        The smallest number of possibilities in any square, not including zero (solved)
        """
        possibility_counts = self.possibility_counts()
        return min(filter(lambda p: p > 0 and possibility_counts[p] > 0, possibility_counts.keys()))
        
    def resolve_pairs(self, possible: npt.NDArray):
            
        has_two_boolmap: npt.NDArray = sum(possible) == 2  # 9x9 array # type: ignore
        for row, col in product(range(self.height), range(self.width)):
            if not has_two_boolmap[row, col]:
                continue
            
            interacting = self.get_interacting_indicies(row, col)  # all ones in the same row, col, or box
            for irow, icol in interacting:
                if all(possible[irow, icol] == possible[row, col]):  # has the same two possibilities
                    contexts = list(self.get_interaction_context((row, col), (irow, icol)))
                    for context in contexts:
                        match context:
                            case InteractionContext.ROW:
                                indicies = set(product([row] * 9, range(9)))
                                
                            case InteractionContext.COL:
                                indicies = set(product(range(9), [col] * 9))
                                
                            case InteractionContext.BOX:
                                base_row, base_col = row - row%3, col - col%3  # top left of block
                                indicies = set(product(range(base_row, base_row+3), range(base_col, base_col+3)))
                        
                        indicies -= {(row, col), (irow, icol)}
                        known_map = possible[row, col]
                        for i, j in indicies:
                            possible[i, j] = (possible[i, j].astype(bool) | known_map.astype(bool)) & ~(possible[i, j].astype(bool) & known_map.astype(bool))
        return possible
    
    def get_row(self, row_n: int):
        return self.board[row_n]
    
    def get_col(self, col_n: int):
        return self.board[:, col_n]
                        
    
    
    def possibility_counts(self) -> dict[int, int]:
        """
        Gets a count of how many squares have each possibility count [0-9]
        """
        possible_key = self.create_possible()
        count = {}
        for row, col in product(range(self.height), range(self.width)):
            possible = possible_key[row, col]
            n_possible = int(sum(possible))
            if n_possible in count:
                count[n_possible] += 1
            else:
                count[n_possible] = 1
        return count
    
        
    def create_possible(self) -> np.ndarray:
        # fmt: off
        possible = np.ones((self.height, self.width, 9))
        given = self.board != 0
        possible[given] = np.zeros((9,))  # any existing spots are set to all false possibles
        
        for row, col in product(range(self.height), range(self.width)):
            if given[row, col]:
                value = self.board[row, col]
                inter = self.get_interacting_indicies(row, col)
                for i, j in inter:
                    possible[i, j, value-1] = False
        
        return possible
        # fmt: on
    
    
    def get_interaction_context(self, a: tuple[int, int], b: tuple[int, int]) -> Generator["InteractionContext", None, None]:
        if a[0] == b[0]:
            yield InteractionContext.ROW
        if a[1] == b[1]:
            yield InteractionContext.COL
        if self.box_n(*a) == self.box_n(*b):
            yield InteractionContext.BOX
    

    @property
    def height(self):
        return len(self.board)

    @property
    def width(self):
        return len(self.board[0])

    @classmethod
    def random(cls, height: int = 9, width: int = 9) -> "Sudoku":
        return cls(np.random.randint(1, 10, (height, width)))

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


class InteractionContext(Enum):
    ROW = 0
    COL = 1
    BOX = 2

sudoku = Sudoku(medium_1) 
sudoku.solve()
# find 2 boxes with matching pairs of possible values
# that are intersecting
# subtract from other intersecting boxes in the same intersection context (row, col, box)

# then, extend to 3 with matching triplets of possible values
# ...