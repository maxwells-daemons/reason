//! Utilities for loading and running the FFO engame suite.

use crate::board::{Board, Location};
use core::panic;
use std::fs::File;
use std::io;
use std::io::prelude::*;

pub struct FFOPosition {
    board: Board,
    best_move: Option<Location>,
    score: i8,
}

pub fn load_ffo_positions(path: &str) -> Vec<FFOPosition> {
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);

    reader
        .lines()
        .map(|line| parse_ffo_position(line.unwrap()))
        .collect()
}

fn parse_ffo_position(ffo_string: String) -> FFOPosition {
    let mut sections = ffo_string.split_whitespace();

    let board_str = sections.next().unwrap();
    let player = sections.next().unwrap();

    let mv = match sections.next().unwrap() {
        "-1" => None,
        n => Some(Location::from_index(n.parse().unwrap())),
    };
    let score = sections.next().unwrap().parse().unwrap();

    FFOPosition {
        board: parse_ffo_board(board_str, player),
        best_move: mv,
        score: score,
    }
}

fn parse_ffo_board(board_str: &str, player: &str) -> Board {
    let mut black_bitboard: u64 = 0;
    let mut white_bitboard: u64 = 0;

    for char in board_str.chars() {
        black_bitboard <<= 1;
        white_bitboard <<= 1;

        if char == 'O' {
            white_bitboard |= 1;
        } else if char == 'X' {
            black_bitboard |= 1;
        } else if char != '-' {
            panic!("Unknown character in FFO bitboard: {}", char)
        }
    }

    match player {
        "Black" => Board {
            player_bitboard: black_bitboard,
            opponent_bitboard: white_bitboard,
        },
        "White" => Board {
            player_bitboard: white_bitboard,
            opponent_bitboard: black_bitboard,
        },
        player => panic!("FFO player is not Black or White: {}", player),
    }
}
