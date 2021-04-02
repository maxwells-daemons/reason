use crate::{Game, Move, ParseMoveError};

/// Play an interactive Othello game.
pub fn play_interactive() {
    use std::io::Write;
    let mut game = Game::default();
    let mut mv: Move;

    while !game.is_finished() {
        loop {
            println!("\n{}\n", game);

            print!("Enter a move: ");
            std::io::stdout().flush().unwrap();
            let mut input_line = String::new();
            std::io::stdin().read_line(&mut input_line).unwrap();
            let parsed_mv: Result<Move, ParseMoveError> =
                input_line.strip_suffix('\n').unwrap().parse();

            if parsed_mv.is_ok() {
                mv = parsed_mv.unwrap();
            } else {
                println!("Cannot parse move.");
                continue;
            }

            let next_state = game.apply_move(mv);
            if next_state.is_ok() {
                game = next_state.unwrap();
                break;
            } else if game.board.get_moves().is_empty() {
                println!("Invalid move. Please enter 'pass'.");
                continue;
            } else {
                println!("Invalid move. Legal moves: {}", game.board.get_moves());
                continue;
            }
        }
    }

    if let Some(winner) = game.winner() {
        println!("Winner: {}.", winner);
    } else {
        println!("Draw.")
    }
}
