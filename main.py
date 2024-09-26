from enum import Enum
from typing import Callable, Generator
import numpy as np
import numpy.typing as npt
from itertools import product  # itertools.product my beloved
from puzzles import *
from copy import deepcopy


class Sudoku:
    def __init__(self, board: list[list[int]] | npt.NDArray[np.int8]) -> None:
        self.board: np.ndarray = np.array(board)
        self.box_height: int = 3
        self.box_width: int = 3

    def is_filled(self) -> bool:
        """
        Whether or not all spots in the board have an answer placed
        """
        return all(all(row) for row in self.board)

    def is_valid(self) -> bool:
        for row, col in product(range(self.height), range(self.width)):
            value = self.board[row, col]
            inter = self.get_interacting_indicies(row, col)
            for i, j in inter:
                if self.board[i, j] == value:
                    return False

        return True
    
    def is_complete(self):
        return self.is_filled() and self.is_valid()
    

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
        rows = {(row, col_t) for col_t in range(0, self.width)}
        cols = {(row_t, col) for row_t in range(0, self.height)}
        box = self.box_group(row, col)

        return (rows | cols | box) - {(row, col)}

    def solve(self):
        max_runs = 20
        current_runs = 0
        while current_runs < max_runs and not self.is_filled():
            current_runs += 1
            
            self.singles_solve()
            print(self, "single")
            
            possible = self.resolve_pairs(self.create_possible())
            self.replace_singles(possible)
            print(self, "pairs")
            
            possible = self.resolve_triples(possible)
            self.replace_singles(possible)
            print(self, "triple")
            
            print(self.possibility_counts())
            

    def singles_solve(self):
        while not self.is_filled():
            old = deepcopy(self.board)
            self.replace_singles(self.create_possible())
            if np.all(old == self.board):
                return

    def replace_singles(self, possible: np.ndarray) -> None:
        """
        Replaces all apparent values in .board using the possibility map given. Only does one pass
        """
        for row, col in product(range(self.height), range(self.width)):
            pk = possible[row, col]
            if sum(pk) == 1:
                value = np.where(pk == 1)[0][0] + 1  # index + 1
                possible[row, col] = np.zeros((9,))
                self.board[row, col] = value

    def least_unresolved(self) -> int:
        """
        The smallest number of possibilities in any square, not including zero (solved)
        """
        possibility_counts = self.possibility_counts()
        return min(
            filter(
                lambda p: p > 0 and possibility_counts[p] > 0, possibility_counts.keys()
            )
        )

    def resolve_pairs(self, possible: npt.NDArray):

        pair_boolmap = self.create_has_n_boolmap(2)
        for row, col in product(range(self.height), range(self.width)):
            if not pair_boolmap[row, col]:
                continue
            
            known_map = possible[row, col].astype(bool)

            interacting = self.get_interacting_indicies(
                row, col
            )  # all ones in the same row, col, or box
            for irow, icol in interacting:
                if not all(possible[irow, icol] == possible[row, col]):
                    continue
                
                contexts = list(self.get_interaction_context((row, col), (irow, icol)))
                # print(f"found pair: [{(row, col), (irow, icol)}] ctx: {contexts} map: {possible[irow, icol]}")
                for context in contexts:
                    match context:
                        case InteractionContext.ROW:
                            indicies = {(row, i_col) for i_col in range(9)}

                        case InteractionContext.COL:
                            indicies = {(i_row, col) for i_row in range(9)}

                        case InteractionContext.BOX:
                            base_row, base_col = (
                                row - row % 3,
                                col - col % 3,
                            )
                            indicies = set(
                                product(
                                    range(base_row, base_row + 3),
                                    range(base_col, base_col + 3),
                                )
                            )

                    indicies -= {(row, col), (irow, icol)}
                    # print(f"subtracting from {len(indicies)} idx", indicies)
                    for i, j in indicies:
                        # [1, 1, 0, 0, 0, 0, 1, 0]  # < something else in the same row "this_possible"
                        # [0, 1, 1, 0, 0, 0, 0, 0]  # < trying to remove "known_map"

                        # [1, 0, 0, 0, 0, 0, 1, 0] $ desired
                        # [1, 0, 1, 0, 0, 0, 1, 0] # xor
                        #        ^
                        this_possible = possible[i, j].astype(bool)
                        possible[i, j] = (
                            (this_possible | known_map)
                            & ~(this_possible & known_map)
                        ) & this_possible
        return possible

    def resolve_triples(self, possible: npt.NDArray):
        triple_boolmap = self.create_has_n_boolmap(3)
        
        
        for row, col in product(range(self.height), range(self.width)):
            if not triple_boolmap[row, col]:
                continue
            
            known_map = possible[row, col].astype(bool)
            interacting = self.get_interacting_indicies(row, col)
            
            indexes = self.find_with_possible(
                possible, 
                known_map, 
                2, 
                req=lambda trow, tcol: 
                    (trow, tcol) in interacting and 
                    (trow, tcol) != (row, col)
            )
            if indexes is None:  # no other two values
                continue
            else:
                a, b = indexes
                
            contexts = list(self.get_interaction_context((row, col), a, b))
            
            for context in contexts:
                match context:
                    case InteractionContext.ROW:
                        indicies = {(row, i_col) for i_col in range(9)}

                    case InteractionContext.COL:
                        indicies = {(i_row, col) for i_row in range(9)}

                    case InteractionContext.BOX:
                        base_row, base_col = (
                            row - row % 3,
                            col - col % 3,
                        )
                        indicies = set(
                            product(
                                range(base_row, base_row + 3),
                                range(base_col, base_col + 3),
                            )
                        )
                
                indicies -= {(row, col), a, b}
                for i, j in indicies:
                    this_possible = possible[i, j].astype(bool)
                    possible[i, j] = (
                        (this_possible | known_map)
                        & ~(this_possible & known_map)
                    ) & this_possible
        
        return possible
            
    
    def find_with_possible(self, possible: npt.NDArray, search_possible: npt.NDArray, n: int, *, req: Callable[[int, int], bool] = lambda r, c: True):
        """
        Returns the first n indexes of `possible` which match `search_possible`, or None if none exist
        """
        cur = []
        for row, col in product(range(self.height), range(self.width)):
            if all(possible[row, col] == search_possible) and req(row, col):
                cur.append((row, col))
                if len(cur) >= n:
                    return cur
                
        return None
            
            
                

    def create_has_n_boolmap(self, n: int):
        possible = self.create_possible()
        start = np.zeros_like(self.board)  # eg. 9x9

        for row, col in product(range(self.height), range(self.width)):
            possible_list = list(
                possible[row, col]
            )  # 9-value boolean-like array for each digit's possibility of being in this square
            if sum(possible_list) == n:
                start[row, col] = True

        return start
        

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
        possible = np.ones((self.height, self.width, 9))
        given = self.board != 0
        possible[given] = np.zeros(
            (9,)
        )  # any existing spots are set to all false possibles

        for row, col in product(range(self.height), range(self.width)):
            if given[row, col]:
                value = self.board[row, col]

                inter = self.get_interacting_indicies(row, col)
                for i, j in inter:
                    possible[i, j, value - 1] = False

        return possible

    def get_interaction_context(
        self, *points: tuple[int, int]
    ) -> Generator["InteractionContext", None, None]:
        if all(a[0] == points[0][0] for a in points):
            yield InteractionContext.ROW
        if all(a[1] == points[0][1] for a in points):
            yield InteractionContext.COL
        if all(self.box_n(*a) == self.box_n(*points[0]) for a in points):
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


sudoku = Sudoku(hard_2)
print(sudoku)
sudoku.solve()
print(sudoku)
print(sudoku.is_valid())
