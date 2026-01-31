import numpy as np
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

ROWS = 6
COLS = 7
EMPTY = "."

P1 = "X"
P2 = "O"

@dataclass
class MoveResult:
    row: int
    col: int
    winner: Optional[str] = None
    draw: bool = False

class Connect4():
    def __init__(self, rows: int = ROWS, cols: int = COLS):
        self.rows = rows
        self.cols = cols
        self.board: List[List[str]] = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        self.last_move: Optional[Tuple[int,int]] = None

    def reset(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                self.board[r][c] = EMPTY 
        self.last_move = None

    def render(self) -> None:
        print("\n  " +  " ".join(str(c + 1) for c in range(self.cols)))
        for r in range(self.rows):
            print("\n  " +  " ".join(self.board[r]))
        print()

    def valid_moves(self) -> List[int]:
        return [c for c in range(self.cols) if self.board[0][c] == EMPTY]

    def drop(self, col: int, token: str) -> Optional[MoveResult]:
        if col < 0 or col >= self.cols:
            return None

        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = token
                self.last_move = (r, col)
                winner = token if self.check_winner(r, col, token) else None
                draw = (winner is None) and (len(self.valid_moves()) == 0)
                return MoveResult(row=r, col=col, winner=winner, draw=draw)

        return None

    def check_winner(self, r: int, c: int, token: str) -> bool:
        # count connnected tokens in each directional pair
        directions = [
            (0,1), # horiz
            (1,0),  # vert
            (1,1),  # diag \
            (1,-1), # diag /
        ]
        for dr, dc in directions:
            if 1 + self._count_dir(r, c, dr, dc, token) + self._count_dir(r, c, -dr, -dc, token) >= 4:
                return True
        return False
        
    def _count_dir(self, r: int, c: int, dr: int, dc: int, token: str) -> int:
        count = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < self.rows and 0 <= cc < self.cols and self.board[rr][cc] == token:
            count +=1
            rr += dr
            cc += dc
        return count

    def clone(self) -> "Connect4":
        g = Connect4(self.rows, self.cols)
        g.board = [row[:] for row in self.board]
        g.last_move = self.last_move
        return g

def choose_ai_move(game: Connect4, ai_token: str, human_token: str) -> int:
    """
    Simple heuristic AI:
    1) If AI can win now, do it
    2) Else if human can win next, block it
    3) Else pick a move biased toward center columns
    """
    valid = game.valid_moves()

    # 1) Winning move
    for c in valid:
        g2 = game.clone()
        res = g2.drop(c, ai_token)
        if res and res.winner == ai_token:
            return c

    # 2) Block human winning move
    for c in valid:
        g2 = game.clone()
        res = g2.drop(c, human_token)
        if res and res.winner == human_token:
            return c

    # 3) Center bias
    center = game.cols // 2
    scored = []
    for c in valid:
        score = -abs(c - center)
        # small random jitter to avoid always identical play
        score += random.random() * 0.01
        scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1]


def prompt_int(prompt: str) -> int:
    while True:
        s = input(prompt).strip()
        try:
            return int(s)
        except ValueError:
            print("Please enter a number.")

def main() -> None:
    
    print("Connect 4 (terminal)")
    print("Drop tokens into columns 1-7. First to connect 4 wins.\n")
    
    game = Connect4()
    
    mode = None
    while mode not in (1, 2):
        mode = prompt_int("Choose mode: 1 = Human vs Human, 2 = Human vs Computer: ")
    
    if mode == 2:
        first = None
        while first not in (1, 2):
            first = prompt_int("Who goes first? 1 = Human (X), 2 = Computer (O): ")
        human_token = P1 if first == 1 else P2
        ai_token = P2 if human_token == P1 else P1
    else:
        human_token = None
        ai_token = None
    
    current = P1
    game.render()
    
    while True:
        valid = game.valid_moves()
        if not valid:
            print("Draw. Board is full.")
            break

        if mode == 2 and current == ai_token:
            col = choose_ai_move(game, ai_token, human_token)
            print(f"Computer plays column {col + 1}")
        else:
            while True:
                col_in = prompt_int(f"Player {current}, choose a column (1-{game.cols}): ")
                col = col_in - 1
                if col in valid:
                    break
                print("Invalid move. Try a non-full column in range.")

        res = game.drop(col, current)
        if res is None:
            print("Illegal move. Try again.")
            continue

        game.render()

        if res.winner:
            if mode == 2 and res.winner == ai_token:
                print("Computer wins.")
            else:
                print(f"Player {res.winner} wins.")
            break
        if res.draw:
            print("Draw.")
            break

        current = P2 if current == P1 else P1


if __name__ == "__main__":
    main()

        