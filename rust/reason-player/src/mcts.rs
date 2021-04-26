//! Code for monte-carlo tree search, used for earlygame and midgame play.

use crate::network::{self, Prediction};
use dot;
use reason_othello::{Game, Move, Player};
use std::collections::HashMap;
use tch::CModule;

pub struct MCTS<'a> {
    root: Game,
    positions: HashMap<Game, PositionData>,
    model: &'a CModule,
    prior_weight: f64,
}

// TODO: track known predecessors for graph backup
#[derive(Debug)]
pub enum PositionData {
    Internal(InternalPosition),
    Leaf(f64),
}

#[derive(Debug)]
pub struct InternalPosition {
    actions: Vec<ActionData>,
    prediction: Prediction,
}

#[derive(Copy, Clone, Debug)]
pub struct ActionData {
    action: Move,
    result: Game,

    visits: i32,
    total_value: f64,
    prior_prob: f64,
}

impl<'a> MCTS<'a> {
    pub fn new(model: &'a CModule, prior_weight: f64) -> Self {
        MCTS {
            root: Game::default(),
            positions: HashMap::new(),
            model,
            prior_weight,
        }
    }

    /// Advance the MCTS tree by a single "simulation".
    pub fn step(&mut self) {
        // Select a leaf of the current MCTS tree by UCB.
        let (leaf, trajectory, previously_visited) = self.select_leaf();

        // Expand the leaf (unless it's a game-ending state we already visited).
        if !previously_visited {
            let expanded = self.expand(leaf);
            self.positions.insert(leaf, expanded);
        }
        let leaf_data = self.positions.get(&leaf).unwrap();

        // Compute the value of the position for the Black player.
        let black_value = if leaf.active_player == Player::Black {
            leaf_data.value()
        } else {
            -1.0 * leaf_data.value()
        };

        // Update the statistics up this trajectory for both players.
        self.update_statistics(trajectory, black_value);
    }

    /// Pick a leaf node according to UCB tree traversal.
    /// Leaves are typically unexplored states, but may also be game-ending states.
    /// Returns: (leaf, trajectory, is_previously_visited).
    fn select_leaf(&self) -> (Game, Vec<usize>, bool) {
        let mut position = self.root;
        let mut trajectory: Vec<usize> = Vec::new();

        // Walk the tree of internal nodes by action until we hit a leaf or new node
        while let Some(position_data) = self.positions.get(&position) {
            if let PositionData::Internal(internal_data) = position_data {
                // Internal case: we've seen this node before and have actions out of it.
                let action_idx = self.ucb_action_index(internal_data);
                trajectory.push(action_idx);
                position = internal_data.actions[action_idx].result;
            } else {
                // Leaf case: UCB led us to a game-ending state and we can't continue.
                return (position, trajectory, true);
            }
        }

        // Board no longer in the tree: this is an unexplored node
        (position, trajectory, false)
    }

    /// Expand a newly-visited position into a node.
    fn expand(&self, position: Game) -> PositionData {
        if position.is_finished() {
            let value = value_finished_game(position);
            return PositionData::Leaf(value);
        }

        let moves = position.get_moves();
        let prediction = network::predict(position.board, moves, self.model);

        let actions = if moves.is_empty() {
            // No legal moves: `actions` just contains the pass move.
            // NOTE: dummy 0.0 prior; this is the only action so UCB doesn't matter.
            let pass_action = ActionData::new(position, Move::Pass, 0.0);
            vec![pass_action]
        } else {
            moves
                .map(|loc| ActionData::new(position, Move::Piece(loc), prediction.policy_at(loc)))
                .collect()
        };

        PositionData::Internal(InternalPosition {
            actions,
            prediction,
        })
    }

    /// Walk along the actions in `trajectory` and update statistics for each,
    /// based on the simulation result at its end.
    fn update_statistics(&mut self, trajectory: Vec<usize>, black_value: f64) {
        let mut position = self.root;

        for move_idx in trajectory {
            let node_data = self.positions.get_mut(&position).unwrap();

            // Compute the value of the leaf for the active player
            let value = if position.active_player == Player::Black {
                black_value
            } else {
                -1.0 * black_value
            };

            // Update the action-value toward the leaf value (for the active player)
            if let PositionData::Internal(internal) = node_data {
                let action = &mut internal.actions[move_idx];
                action.visits += 1;
                action.total_value += value;

                // Proceed to next position
                position = action.result;
            } else {
                panic!("Leaf node reached during backup!")
            }
        }
    }

    /// Get the index of the UCB-suggested action out of a position.
    fn ucb_action_index(&self, position: &InternalPosition) -> usize {
        assert_ne!(position.actions.len(), 0, "Cannot get action for a leaf.");

        let num_visits = position.num_visits();
        let mut best_index = usize::MAX;
        let mut best_score: f64 = f64::NEG_INFINITY;
        for (index, action) in position.actions.iter().enumerate() {
            let score = action.ucb_score(self.prior_weight, num_visits);

            if score > best_score {
                best_index = index;
                best_score = score;
            }
        }

        best_index
    }
}

impl PositionData {
    /// Get the predicted value of a position from the perspective of the active player.
    /// For leaves, +1/-1/0 for win/loss/draw.
    /// For internal nodes, this is the value network's prediction.
    fn value(&self) -> f64 {
        match self {
            PositionData::Internal(internal) => internal.prediction.value,
            PositionData::Leaf(value) => *value,
        }
    }
}

impl InternalPosition {
    /// Get the number of times we've evaluated this node or a descendant.
    fn num_visits(&self) -> i32 {
        let child_visits: i32 = self.actions.iter().map(|action| action.visits).sum();
        // NOTE: counting the parent like this is a departure from AlphaZero, but if we don't do
        // it then the prior isn't used for the first action out of a node, which seems wrong.
        child_visits + 1
    }
}

impl ActionData {
    fn new(parent: Game, action: Move, prior_prob: f64) -> Self {
        let result = parent.apply_move(action).unwrap();

        Self {
            action,
            result,
            visits: 0,
            total_value: 0.0,
            prior_prob,
        }
    }

    /// Compute the average action-value from the perspective of the active player.
    fn average_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / (self.visits as f64)
        }
    }

    /// Compute the UCB score from the perspective of the active player.
    fn ucb_score(&self, prior_weight: f64, parent_visits: i32) -> f64 {
        let prior_weight =
            prior_weight * (parent_visits as f64).sqrt() / ((1 + self.visits) as f64);
        self.average_value() + prior_weight * self.prior_prob
    }
}

/// Compute the value of a finished game: 1/-1/0 for win/loss/draw.
fn value_finished_game(game: Game) -> f64 {
    assert!(game.is_finished());

    match game.winner() {
        None => 0.0,
        Some(player) if player == game.active_player => 1.0,
        _ => -1.0,
    }
}

// Graphviz code for debugging
impl<'a> dot::Labeller<'a, Game, (Game, ActionData)> for MCTS<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("MCTS").unwrap()
    }

    fn node_id(&'a self, node: &Game) -> dot::Id<'a> {
        let id = format!(
            "B{}_{}_{}_{}",
            u64::from(node.board.active_bitboard),
            u64::from(node.board.opponent_bitboard),
            node.active_player,
            node.board.just_passed
        );
        dot::Id::new(id).unwrap()
    }

    fn node_label(&'a self, node: &Game) -> dot::LabelText<'a> {
        let board_label = dot::LabelText::label(node.to_string());
        match self.positions.get(node) {
            None => board_label,
            Some(PositionData::Leaf(value)) => {
                board_label.prefix_line(dot::LabelText::label(format!("Leaf value: {}", value)))
            }
            Some(PositionData::Internal(data)) => {
                let value = data.prediction.value;
                board_label.prefix_line(dot::LabelText::label(format!(
                    "Predicted value: {:.3}",
                    value
                )))
            }
        }
    }

    fn node_color(&'a self, node: &Game) -> Option<dot::LabelText<'a>> {
        match self.positions.get(node) {
            None => Some(dot::LabelText::label("crimson")),
            Some(PositionData::Leaf(_)) => Some(dot::LabelText::label("deepskyblue")),
            _ => None,
        }
    }

    fn edge_label(&'a self, edge: &(Game, ActionData)) -> dot::LabelText<'a> {
        let parent_visits =
            if let PositionData::Internal(data) = self.positions.get(&edge.0).unwrap() {
                data.num_visits()
            } else {
                panic!("Found an edge whose source is not an internal node.")
            };

        dot::LabelText::label(format!(
            "{}\nVisits: {}\nPrior: {:.3}\nAverage value: {:.3}\nUCB score: {:.3}",
            edge.1.action,
            edge.1.visits,
            edge.1.prior_prob,
            edge.1.average_value(),
            edge.1.ucb_score(self.prior_weight, parent_visits)
        ))
    }
}

impl<'a> dot::GraphWalk<'a, Game, (Game, ActionData)> for MCTS<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Game> {
        let mut nodes: Vec<Game> = Vec::new();

        for (&node, data) in self.positions.iter() {
            nodes.push(node);
            if let PositionData::Internal(internal_data) = data {
                for action in internal_data.actions.iter() {
                    nodes.push(action.result);
                }
            }
        }

        nodes.dedup();
        std::borrow::Cow::Owned(nodes)
    }

    fn edges(&'a self) -> dot::Edges<'a, (Game, ActionData)> {
        let mut edges: Vec<(Game, ActionData)> = Vec::new();
        for (&node, data) in self.positions.iter() {
            if let PositionData::Internal(internal_data) = data {
                for action in internal_data.actions.iter() {
                    edges.push((node, *action))
                }
            }
        }
        std::borrow::Cow::Owned(edges)
    }

    fn source(&'a self, edge: &(Game, ActionData)) -> Game {
        edge.0
    }

    fn target(&'a self, edge: &(Game, ActionData)) -> Game {
        edge.1.result
    }
}
