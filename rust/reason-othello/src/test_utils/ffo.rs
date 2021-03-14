//! Utilities for loading and running the FFO engame suite.

use crate::bitboard::Bitboard;
use crate::game::{Board, GameState, Move, Player};
use core::panic;
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[derive(Clone, Copy)]
pub struct FFOPosition {
    pub game_state: GameState,
    pub best_move: Move,
    pub score: i8,
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
    let player = sections.next().unwrap().to_string().parse().unwrap();

    let mv = match sections.next().unwrap() {
        "-1" => Move::PASS,
        n => Move::from_index(n.parse().unwrap()),
    };
    let mut score = sections.next().unwrap().parse().unwrap();

    if player == Player::White {
        score *= -1;
    }

    FFOPosition {
        game_state: GameState {
            board: parse_ffo_board(board_str, player),
            just_passed: false,
        },
        best_move: mv,
        score,
    }
}

fn parse_ffo_board(board_str: &str, player: Player) -> Board {
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

    Board::from_color_bitboards(Bitboard(black_bitboard), Bitboard(white_bitboard), player)
}
