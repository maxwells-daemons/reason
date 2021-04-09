#![feature(exact_size_is_empty)]

use glob;
use itertools::Itertools;
use ndarray::{Array, Array2};
use ndarray_npy;
use reason_othello::{Game, Location, Move, Player};
use std::convert::TryFrom;
use std::fs;
use std::io::{self, BufRead, Read, Seek, SeekFrom};

const LOGISTELLO_INPUT_PATH: &str = "../resources/logistello/logbook.gam";
const LOGISTELLO_OUTPUT_PATH: &str = "../resources/preprocessed/logistello.npy";

const WTHOR_INPUT_GLOB: &str = "../resources/wthor/*.wtb";
const WTHOR_OUTPUT_PATH: &str = "../resources/preprocessed/wthor.npy";

const WTHOR_DB_HEADER_BYTES: u64 = 16;
const WTHOR_GAME_BYTES: usize = 68;
const WTHOR_GAME_HEADER_BYTES: usize = 8;

fn parse_logistello_game(game_str: &str) -> Vec<(Game, Location, i8)> {
    let mut splits = game_str.split_ascii_whitespace();
    let game_str = splits.next().unwrap();
    let score_str = splits.next().unwrap();
    let score: i8 = score_str.parse().unwrap();

    let mut states_and_moves: Vec<(Game, Location)> = Vec::new();
    let mut state = Game::default();

    for mut move_chars in &game_str.chars().take_while(|c| *c != ':').chunks(3) {
        let player = match move_chars.next().unwrap() {
            '+' => Player::Black,
            '-' => Player::White,
            _ => panic!("Unrecognized player symbol"),
        };

        // Ignore pass moves
        if player != state.active_player {
            state = state.apply_move(Move::Pass).unwrap();
        };

        let loc: Location = move_chars.collect::<String>().parse().unwrap();
        states_and_moves.push((state, loc));
        state = state.apply_move(Move::Piece(loc)).unwrap();
    }

    states_and_moves
        .into_iter()
        .map(|(g, m)| {
            (
                g,
                m,
                match g.active_player {
                    Player::Black => score,
                    _ => -1 * score,
                },
            )
        })
        .collect()
}

fn parse_wthor_coords(move_byte: u8) -> (i8, i8) {
    let col = ((move_byte % 10) as i8) - 1;
    let row = ((move_byte / 10) as i8) - 1;
    (row, col)
}

fn parse_wthor_game(game_data: &[u8]) -> Vec<(Game, Location, i8)> {
    assert_eq!(game_data.len(), WTHOR_GAME_BYTES);

    let mut states_and_moves: Vec<(Game, Location)> = Vec::new();
    let mut state = Game::default();

    for &byte in game_data.iter().skip(WTHOR_GAME_HEADER_BYTES) {
        let (row, col) = parse_wthor_coords(byte);

        // Game's over
        if (row == -1) && (col == -1) {
            break;
        }

        // No moves: this player must pass
        if state.board.get_moves().is_empty() {
            state = state.apply_move(Move::Pass).unwrap();
        }

        let loc =
            Location::from_coords(usize::try_from(row).unwrap(), usize::try_from(col).unwrap());
        states_and_moves.push((state, loc));
        state = state.apply_move(Move::Piece(loc)).unwrap();
    }

    let black_score = state.score_absolute_difference(Player::Black);

    states_and_moves
        .into_iter()
        .map(|(g, m)| {
            (
                g,
                m,
                match g.active_player {
                    Player::Black => black_score,
                    _ => -1 * black_score,
                },
            )
        })
        .collect()
}

fn make_datapoint(inputs: (Game, Location, i8)) -> Vec<u64> {
    let (game, loc, score) = inputs;
    vec![
        game.board.active_bitboard.into(),
        game.board.opponent_bitboard.into(),
        (u64::try_from(i64::from(score) + 64)).unwrap(),
        loc.to_onehot().into(),
    ]
}

fn main() -> io::Result<()> {
    println!("Reading Logistello data from {}", LOGISTELLO_INPUT_PATH);
    let logistello_file = fs::File::open(LOGISTELLO_INPUT_PATH)?;
    let logistello_data = io::BufReader::new(logistello_file)
        .lines()
        .map(|x| parse_logistello_game(&x.unwrap()))
        .concat();

    println!("Compressing Logistello data.");
    let mut logistello_arr = Array2::<u64>::zeros((logistello_data.len(), 4));
    for (datapoint, mut row) in logistello_data.into_iter().zip(logistello_arr.rows_mut()) {
        row.assign(&Array::from_vec(make_datapoint(datapoint)));
    }

    println!("Writing Logistello data to {}", LOGISTELLO_OUTPUT_PATH);
    ndarray_npy::write_npy(LOGISTELLO_OUTPUT_PATH, &logistello_arr).unwrap();
    println!(
        "Wrote {} examples of Logistello data.",
        logistello_arr.shape()[0]
    );

    println!("Reading WTHOR data from {}", WTHOR_INPUT_GLOB);
    let mut wthor_data: Vec<(Game, Location, i8)> = Vec::new();
    let wthor_paths = glob::glob(WTHOR_INPUT_GLOB).unwrap();
    for path in wthor_paths {
        let mut wthor_file = fs::File::open(path.unwrap()).unwrap();

        // Ignore the DB header
        wthor_file
            .seek(SeekFrom::Start(WTHOR_DB_HEADER_BYTES))
            .unwrap();

        // Read the file in one pass
        let mut file_bytes = Vec::new();
        wthor_file.read_to_end(&mut file_bytes).unwrap();

        for game_bytes in file_bytes.chunks(WTHOR_GAME_BYTES) {
            wthor_data.append(&mut parse_wthor_game(game_bytes));
        }
    }

    println!("Compressing WTHOR data.");
    let mut wthor_arr = Array2::<u64>::zeros((wthor_data.len(), 4));
    for (datapoint, mut row) in wthor_data.into_iter().zip(wthor_arr.rows_mut()) {
        row.assign(&Array::from_vec(make_datapoint(datapoint)));
    }

    println!("Writing WTHOR data to {}", WTHOR_OUTPUT_PATH);
    ndarray_npy::write_npy(WTHOR_OUTPUT_PATH, &wthor_arr).unwrap();
    println!("Wrote {} examples of WTHOR data.", wthor_arr.shape()[0]);

    println!("Done!");

    Ok(())
}
