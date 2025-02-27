# Low-Level Design (LLD) for Chess Game

## 1. Objective

Implement a two-player chess game with all standard rules.

## 2. Key Considerations

- Object-oriented design for chess pieces and board.
- Managing valid moves for each piece.
- Tracking the game state (check, checkmate, stalemate).
- Supporting move validation and turn-based gameplay.
- Providing a user interface (CLI, GUI, or web-based).

## 3. Class Design

### 3.1 Key Classes

```java
class ChessGame {
    private Board board;
    private Player playerWhite;
    private Player playerBlack;
    private Player currentTurn;
    private GameStatus status;

    public void startGame();
    public boolean makeMove(Move move);
    public boolean isCheckmate();
    public boolean isStalemate();
}

class Board {
    private Cell[][] cells;

    public void initializeBoard();
    public boolean isMoveValid(Move move);
    public void updateBoard(Move move);
    public boolean isKingInCheck(Player player);
}

class Cell {
    private int row;
    private int column;
    private Piece piece;
}

abstract class Piece {
    protected PieceType type;
    protected Player owner;
    protected boolean isCaptured;

    public abstract boolean isValidMove(Move move, Board board);
}

class King extends Piece {
    public boolean isValidMove(Move move, Board board);
}

class Queen extends Piece {
    public boolean isValidMove(Move move, Board board);
}

class Rook extends Piece {
    public boolean isValidMove(Move move, Board board);
}

class Bishop extends Piece {
    public boolean isValidMove(Move move, Board board);
}

class Knight extends Piece {
    public boolean isValidMove(Move move, Board board);
}

class Pawn extends Piece {
    public boolean isValidMove(Move move, Board board);
}

enum PieceType {
    KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN
}

enum GameStatus {
    ACTIVE, CHECKMATE, STALEMATE, DRAW
}

class Move {
    private Cell start;
    private Cell end;
    private Piece movedPiece;
    private Piece capturedPiece;
}

class Player {
    private String name;
    private boolean isWhite;
}
```

## 4. Database Schema (For Persistent Storage)

### 4.1 Players Table

```sql
CREATE TABLE Players (
    player_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    is_white BOOLEAN
);
```

### 4.2 Games Table

```sql
CREATE TABLE Games (
    game_id VARCHAR PRIMARY KEY,
    player_white VARCHAR REFERENCES Players(player_id),
    player_black VARCHAR REFERENCES Players(player_id),
    status ENUM('ACTIVE', 'CHECKMATE', 'STALEMATE', 'DRAW')
);
```

### 4.3 Moves Table

```sql
CREATE TABLE Moves (
    move_id SERIAL PRIMARY KEY,
    game_id VARCHAR REFERENCES Games(game_id),
    player_id VARCHAR REFERENCES Players(player_id),
    start_position VARCHAR,
    end_position VARCHAR,
    piece_moved VARCHAR,
    piece_captured VARCHAR,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 5. System Workflow

### 5.1 Game Flow

1. **Game Initialization:** Players are assigned white and black.
2. **Board Setup:** Pieces are placed in their initial positions.
3. **Player Moves:** Each player takes turns making moves.
4. **Move Validation:** The system checks if the move is valid.
5. **Check Detection:** If a king is in check, the player must respond.
6. **Game End Conditions:** Checkmate, stalemate, or draw are detected.

## 6. Optimizations and Trade-offs

- **Efficient Move Validation:** Use a **bitboard representation** for faster move lookups.
- **State Management:** Store **game state in-memory** for active games to reduce database calls.
- **AI/Automation:** Implement a **bot player** for single-player mode.
- **Concurrency Handling:** Ensure proper turn management in a multi-user environment.

## 7. Conclusion

This LLD outlines a structured way to implement a **Chess Game**, covering class design, database schema, and system workflows. The design ensures a robust and maintainable approach to handling game logic and state management.
