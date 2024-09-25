import numpy as np
from itertools import product


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
        rows = {(row, col_t) for col_t in range(0, self.width)}
        cols = {(row_t, col) for row_t in range(0, self.height)}
        box = self.box_group(row, col)

        return (rows | cols | box) - {(row, col)}

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


def solve(puzzle: Sudoku):
    possible = np.ones((puzzle.height, puzzle.width, 9))
    possible[
        puzzle.board[
            :,
            :,
        ]
        != 0
    ] = np.zeros(
        (9,)
    )  # any existing spots are set to all false possibles

    symbols = list(range(1, 10))
    print(puzzle.interacting_indicies(3, 3))


suds = Sudoku(
    [
        [3, 0, 0, 2, 0, 1, 0, 0, 0],
        [7, 4, 0, 0, 0, 0, 0, 1, 9],
        [0, 2, 0, 0, 6, 0, 5, 0, 0],
        [0, 3, 0, 7, 4, 0, 0, 0, 1],
        [0, 0, 8, 0, 0, 0, 9, 0, 0],
        [6, 0, 0, 0, 9, 2, 0, 5, 0],
        [0, 0, 2, 0, 8, 0, 0, 4, 0],
        [1, 5, 0, 0, 0, 0, 0, 9, 7],
        [0, 0, 0, 9, 0, 3, 0, 0, 2],
    ]
)
print(suds.board)
solve(suds)
