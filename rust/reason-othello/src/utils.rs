//! Miscellaneous project utilities.

use crate::EDGE_LENGTH;
use std::fmt::{self, Formatter};
use std::iter::Iterator;

/// Format 64 characters into a pretty grid format.
/// `piece_iter` must yield exactly 64 items.
pub fn format_grid<T: Iterator<Item = char>>(mut piece_iter: T, f: &mut Formatter) -> fmt::Result {
    write!(f, "   A B C D E F G H")?;

    for row in 0..EDGE_LENGTH {
        write!(f, "\n {} ", row + 1)?;
        for _ in 0..EDGE_LENGTH {
            write!(f, "{} ", piece_iter.next().ok_or(fmt::Error)?)?;
        }
    }

    match piece_iter.next() {
        None => Ok(()),
        _ => Err(fmt::Error),
    }
}
