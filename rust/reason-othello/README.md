**reason-othello** is a full-featured Othello library backed by a blazing fast bitboard.

Key features include:
 - The fastest bitboard in the West, written with explicit SIMD to use vector instructions. 
 - Three level of abstraction trading off speed and convenience: raw bitboard functions, a fast and unchecked `Board` type for engines, and a safe high-level `Game` type.
 - A C-style FFI for using bitboard functions outside of Rust.

This package was developed as part of the [reason](https://github.com/maxwells-daemons/reason/) engine.