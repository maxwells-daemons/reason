//! Utilities used for testing and benchmarking.

pub mod ffo;

mod perft;
pub use perft::run_perft;

mod play;
pub use play::play_interactive;
