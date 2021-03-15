//! Tests on the FFO endgame suite.

use indicatif::ProgressIterator;
use reason_othello::test_utils::ffo;
use reason_solver;

fn ffo_exact(path: &str) {
    for position in ffo::load_ffo_positions(path).iter().progress() {
        let score = reason_solver::solve_exact(position.game_state);
        assert_eq!(score, position.score);
    }
}

fn ffo_win_loss_draw(path: &str) {
    for position in ffo::load_ffo_positions(path).iter().progress() {
        let score = reason_solver::solve_win_loss_draw(position.game_state);
        assert_eq!(score.signum(), position.score.signum());
    }
}

#[test]
fn ffo_13_14_exact() {
    ffo_exact("../../resources/ffo/13_14.txt")
}

#[test]
fn ffo_13_14_wld() {
    ffo_win_loss_draw("../../resources/ffo/13_14.txt")
}

#[test]
fn ffo_15_16_exact() {
    ffo_exact("../../resources/ffo/15_16.txt")
}

#[test]
fn ffo_15_16_wld() {
    ffo_win_loss_draw("../../resources/ffo/15_16.txt")
}

#[test]
fn ffo_17_18_exact() {
    ffo_exact("../../resources/ffo/17_18.txt")
}

#[test]
fn ffo_17_18_wld() {
    ffo_win_loss_draw("../../resources/ffo/17_18.txt")
}

#[test]
fn ffo_19_20_exact() {
    ffo_exact("../../resources/ffo/19_20.txt")
}

#[test]
fn ffo_19_20_wld() {
    ffo_win_loss_draw("../../resources/ffo/19_20.txt")
}

#[test]
fn ffo_21_22_exact() {
    ffo_exact("../../resources/ffo/21_22.txt")
}

#[test]
fn ffo_21_22_wld() {
    ffo_win_loss_draw("../../resources/ffo/21_22.txt")
}
