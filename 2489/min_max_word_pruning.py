class CrosswordGame:
    def __init__(self):
        self.initialize_game()

    def initialize_game(self):
        # Initialize your game state here
        self.current_state = [
            ['#','_','_','_','_','#','_'], # A
            ['_','#','#','_','#','#','_'], # B
            ['_','#','#','_','#','#','_'], # C
            ['_','#','_','_','_','_','_'], # D
            ['#','#','_','#','#','#','#'], # E
            ['_','_','_','_','_','_','_'], # F
            ['#','#','_','#','#','#','#'], # G
            ['_','_','_','_','_','_','#'], # H
            ['#','#','_','#','#','#','#'], # I
            ['#','#','_','_','_','_','#'], # J
            
        ]
        self.player_turn = 'Player1'
        
        self.scores = {'user': 0, 'computer': 0}
        
        self.word_co_ordinates = {
            'DUCK': [(0,1), (0,2), (0,3), (0,4)],
            'CROW': [(0, 3), (1, 3), (2, 3), (3, 3)],
            'DOVE': [(0, 6), (1, 6), (2, 6), (3, 6)],
            'SWAN': [(3, 2), (3, 3), (3, 4), (3, 5)],
            'PEACOCK': [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
            'SPARROW': [(3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)],
            'PARROT': [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5)],
            'WREN': [(9, 2), (9, 3), (9, 4), (9, 5)],
            'EMU': [(1,0),(2,0),(3,0)]
        }
        
        self.words = list(self.word_co_ordinates.keys())
        self.placed_words = []
        
        
    def get_row(self, key):
        col = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4,
            'F': 5,
            'G': 6,
            'H': 7,
            'I': 8,
            'J': 9
        }
        
        if key not in col:
            raise ValueError(f'Invalid Move: {key}')
        
        return col[key]
    
    def get_column(self, key):
        
        if int(key) not in range(1, 8):
            raise ValueError(f'Invalid Move: {key}')
        
        return int(key) - 1
        
    def validate_input(self, move):
        # Check if the string length is at least 2
        if len(move) < 2:
            return (False, [-1,-1], f'Invalid Move Must follow constraint of 2 characters length: {move}')        
        
        # Check if the second character is between 1 and 7
        if not ('1' <= move[0] <= '7'):
            return (False, [-1,-1], f'Row count must be in range (1,7): {move}')
        
        # Check if the first character is between A and J
        if not ('A' <= move[1] <= 'J'):
            return (False, [-1,-1], f'Column must be in range (A,J): {move}')
        
        row,col = self.get_row(move[1]), self.get_column(move[0])
        
        # Check if the index is allowed to make a move
        if self.current_state[row][col] == '#':
            return (False, [-1,-1], f'Invalid Move Location: {move}. This co-ordinate is not allowed to position a word.')
        
        # If all checks passed, return True
        return (True, [row, col], '')

    def print_current_board(self):
        # Print the board
        print("  1 2 3 4 5 6 7")
        print("  -------------")
        row_label = 'A'
        for row in self.current_state:
            print(f"{row_label}|{' '.join(row)}|")
            row_label = chr(ord(row_label) + 1)  # Next character in the alphabet
        print("  -------------")
        
    
    def make_a_move(self, move, word, isPlayer = True):
        # Make sure senstivity doesn't matter
        word = word.upper()
        if word not in self.word_co_ordinates:
            return (False, self.current_state, f'Invalid Word: {word}. Pick one from {", ".join(self.words)}')
        
        (is_valid, [x,y], error_message) = self.validate_input(move)
        
        if not is_valid:
            return (False, self.current_state, error_message)
        
        [x_init, y_init], *_ = self.word_co_ordinates[word]
        
        if x_init != x or y_init != y:
            prev_score = self.scores['user']
            new_score = self.scores['user'] - 1
            if not isPlayer:
                prev_score = self.scores['computer']
                self.scores['computer'] -= 1
            else:
                self.scores['user'] -= 1
            return (False, self.current_state, f'Invalid Word Placment for {word}: {move} (Expected:({x_init+1},{y_init+1}), Found({x+1},{y+1})). Reduced score from {prev_score} to {new_score}')
        
        self.place_a_word(word)
        return (True, self.current_state, '')
    
    def min_max_algorithm(self,remaining_words, is_maximizing = True, alpha=float('-inf'), beta=float('inf'), depth=0):
        # Base case: No more words left
        # Depth can be printed for tracking recursion depth
        if len(remaining_words) == 0:
            return 0, ""
        
        if is_maximizing:
            max_eval = float('-inf')
            best_word = None
            for word in remaining_words:
                new_remaining = remaining_words[:]
                new_remaining.remove(word)
                eval_to_maximize, _ = self.min_max_algorithm(new_remaining, False, alpha, beta, depth + 1)
                max_eval = max(max_eval, eval_to_maximize + len(word))  # Add current word's length to evaluation
                if max_eval == eval_to_maximize + len(word):
                    best_word = word
                alpha = max(alpha, eval_to_maximize)
                if beta <= alpha:
                    break
            return max_eval, best_word
        else:  # Minimizing player
            min_eval = float('inf')
            worst_word = None
            for word in remaining_words:
                new_remaining = remaining_words[:]
                new_remaining.remove(word)
                eval_to_minimize, _ = self.min_max_algorithm(new_remaining, True, alpha, beta, depth + 1)
                min_eval = min(min_eval, eval_to_minimize)
                if min_eval == eval_to_minimize:
                    worst_word = word
                beta = min(beta, eval_to_minimize)
                if beta <= alpha:
                    break
            # In minimizing turn, we do not add the word's length to evaluation
            return min_eval, worst_word


    def is_valid_move(self, word, position, direction):
        # Convert row letter to index
        row = ord(position[0].upper()) - ord('A')
        col = int(position[1])

        if direction.upper() == 'A':
            # Check if word fits horizontally without going out of bounds
            if col + len(word) > 7:  # 7 is the width of the board
                return False
        elif direction.upper() == 'D':
            # Check if word fits vertically without going out of bounds
            if row + len(word) > 10:  # 10 is the height of the board
                return False
        else:
            # Invalid direction
            return False
        # Further checks for overlaps and exact fit can be added here
        return True


    def is_end_of_game(self):
    # If no more words are available, the game ends
        return len(self.words) == 0
    
    def compute_best_move(self):
        _, best_move = self.min_max_algorithm(self.words)
        return best_move
    
    def player_move(self):
        valid_move = False
        while not valid_move:
            try:
                player_input = input("Enter your move (e.g., '1A SWAN'): ").upper()
                move, word = player_input.split(' ')
                if word not in self.words:
                    print(f"Invalid word. Choose from: {', '.join(self.words)}")
                    continue
                is_valid, _, error_message = self.make_a_move(move, word)
                if is_valid:
                    valid_move = True
                    self.scores['user'] += len(word)  # Adjust scoring as necessary
                    self.words.remove(word)  # Remove the word from the available list
                    print(f"Placed '{word}' at {move}.")
                else:
                    print(error_message)
            except ValueError:
                print("Invalid input. Please enter a move in the format '1A SWAN'.")
                
    def place_a_word(self, word):
        co_ordinates = self.word_co_ordinates[word]
        for [x, y], letter in zip(co_ordinates, word):
            self.current_state[x][y] = letter

    def computer_move(self):
        print("Computer's turn:")
        best_move = self.compute_best_move()
        if best_move:
            self.place_a_word(best_move)
            self.scores['computer'] += len(best_move)  # Adjust scoring as necessary
            self.words.remove(best_move)  # Remove the word from the available list
            print(f"Computer placed '{best_move}'.")
        else:
            print("No valid moves left for the computer.")
            
    def play(self):
        while not self.is_end_of_game():
            self.print_current_board()
            if self.player_turn == 'Player1':
                self.player_move()
                self.player_turn = 'Computer'
            else:
                self.computer_move()
                self.player_turn = 'Player1'
                
        # Game over
        self.print_current_board()
        print("Game over.")
        print(f"Final scores - Player: {self.scores['user']}, Computer: {self.scores['computer']}")
        if self.scores['user'] > self.scores['computer']:
            print("You win!")
        elif self.scores['user'] < self.scores['computer']:
            print("Computer wins!")
        else:
            print("It's a tie!")
            

if __name__ == "__main__":
    game = CrosswordGame()
    game.play()


