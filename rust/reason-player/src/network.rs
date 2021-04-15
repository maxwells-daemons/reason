//! Code for working with the agent neural network.

use tch::{self, Device, IValue, IndexOp, Kind, Tensor};

use reason_othello::{self, bitboard::Bitboard, Board, Location, LocationList};
use std::convert::TryFrom;

// Redefine these with types that `tch` likes
const EDGE_LENGTH: i64 = reason_othello::EDGE_LENGTH as i64;
const NUM_SPACES: i64 = reason_othello::NUM_SPACES as i64;

#[derive(Debug)]
pub struct Prediction {
    // Shape guaranteed to be [8, 8]
    policy: Tensor,
    pub value: f64,
}

impl Prediction {
    pub fn policy_at(&self, loc: Location) -> f64 {
        let (row, col) = loc.to_coords();
        self.policy.i((row as i64, col as i64)).double_value(&[])
    }
}

fn tensorize_plane(bitboard: Bitboard) -> Tensor {
    Tensor::of_slice(&bitboard.unpack()).view_(&[EDGE_LENGTH, EDGE_LENGTH])
}

// TODO: batched featurize
fn featurize(board: Board) -> Tensor {
    // Piece features
    let active_pieces = tensorize_plane(board.active_bitboard).totype(Kind::Float);
    let opponent_pieces = tensorize_plane(board.opponent_bitboard).totype(Kind::Float);

    // Positional features
    let positions = Tensor::linspace(0, 1, EDGE_LENGTH, (Kind::Float, Device::Cpu));
    let x_positions = positions.unsqueeze(0).repeat(&[EDGE_LENGTH, 1]);
    let y_positions = positions.unsqueeze(1).repeat(&[1, EDGE_LENGTH]);

    let corner_mask = Tensor::of_slice(&CORNER_MASK_DATA).reshape(&[EDGE_LENGTH, EDGE_LENGTH]);
    let edge_mask = Tensor::of_slice(&EDGE_MASK_DATA).reshape(&[EDGE_LENGTH, EDGE_LENGTH]);

    Tensor::stack(
        &[
            active_pieces,
            opponent_pieces,
            x_positions,
            y_positions,
            corner_mask,
            edge_mask,
        ],
        0,
    )
}

// TODO: batched predict
pub fn predict(board: Board, legal_moves: LocationList, model: &tch::CModule) -> Prediction {
    tch::no_grad(|| {
        // Introduce dummy batch dimension; shape [#BF, 8, 8] -> [1, #BF, 8, 8]
        let features = featurize(board).unsqueeze_(0);

        // Compute forward pass
        let inputs = [IValue::Tensor(features)];
        let outputs = model.forward_is(&inputs).unwrap();

        let mut output_values = match outputs {
            IValue::Tuple(vec) => vec,
            _ => panic!("Invalid model output type."),
        };

        // Eliminate dummy batch dimension and extract; shape [1] -> f64
        let value = Tensor::try_from(output_values.pop().unwrap())
            .unwrap()
            .double_value(&[0]);

        // Eliminate dummy batch dimension; shape [1, 8, 8] -> [8, 8]
        let mut policy = Tensor::try_from(output_values.pop().unwrap())
            .unwrap()
            .squeeze_();

        // Mask out invalid moves
        let invalid_mask = tensorize_plane(Bitboard::from(legal_moves)).bitwise_not_();
        let _ = policy.masked_fill_(&invalid_mask, f64::NEG_INFINITY);

        // Normalize policy distribution
        let policy = policy
            .view_(&[NUM_SPACES])
            .softmax(0, Kind::Float)
            .view_(&[EDGE_LENGTH, EDGE_LENGTH]);

        Prediction { policy, value }
    })
}

const CORNER_MASK_DATA: [f32; reason_othello::NUM_SPACES] = [
    1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 1.0000,
];

const EDGE_MASK_DATA: [f32; reason_othello::NUM_SPACES] = [
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
    1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
    1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000,
];
