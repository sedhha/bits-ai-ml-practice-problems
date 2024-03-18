def min_max_algorithm(remaining_words, is_maximizing = True, alpha=float('-inf'), beta=float('inf'), depth=0):
    # Base case: No more words left
    if len(remaining_words) == 0:
        return 0, ""
    
    if is_maximizing:
        max_eval = float('-inf')
        best_word = None
        for word in remaining_words:
            new_remaining = remaining_words[:]
            new_remaining.remove(word)
            eval_to_maximize, _ = min_max_algorithm(new_remaining, False, alpha, beta, depth + 1)
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
            eval_to_minimize, _ = min_max_algorithm(new_remaining, True, alpha, beta, depth + 1)
            min_eval = min(min_eval, eval_to_minimize)
            if min_eval == eval_to_minimize:
                worst_word = word
            beta = min(beta, eval_to_minimize)
            if beta <= alpha:
                break
        # In minimizing turn, we do not add the word's length to evaluation
        return min_eval, worst_word


    
    
    



    
if __name__ == "__main__":
    # Test the function
    remaining_words = ["DUCK", "TOLL", "ROUNDING", "TURFER","TURG"]
    score, best_move = min_max_algorithm(remaining_words, is_maximizing=True)
    print(f"Best word to choose: {best_move}, Potential max score: {score}")