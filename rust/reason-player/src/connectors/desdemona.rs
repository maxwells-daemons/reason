//! [`Connector`] for interfacing with [desdemona](https://github.com/cruzsbrian/desdemona).

use super::Connector;
use async_std::io;
use async_trait::async_trait;
use reason_othello::{Game, Location, Move, Player};

pub struct DesdemonaConnector {
    game: Game,
    player: Player,
}

impl DesdemonaConnector {
    pub fn new(player: Player) -> Self {
        Self {
            game: Game::default(),
            player,
        }
    }
}

#[async_trait]
impl Connector for DesdemonaConnector {
    fn set_ready(&mut self) {
        println!("ready");
    }

    fn get_player_color(&self) -> Player {
        self.player
    }

    fn make_move(&mut self, mv: Move) {
        self.game = self.game.apply_move(mv).unwrap();
        match mv {
            Move::Pass => println!("pass"),
            Move::Piece(loc) => {
                let (row, col) = loc.to_coords();
                println!("{} {}", row, col);
            }
        }
    }

    async fn get_opponent_move(&mut self) -> Move {
        let mut buffer = String::new();
        eprintln!("BEFORE OPP MOVE\n{}", self.game);
        io::stdin().read_line(&mut buffer).await.unwrap();

        let mv = if buffer.as_str() == "pass" {
            Move::Pass
        } else {
            let mut segments = buffer.split(" ");
            let row: usize = segments.next().unwrap().parse().unwrap();
            let col: usize = segments.next().unwrap().parse().unwrap();

            Move::Piece(Location::from_coords(row, col))
        };

        self.game = self.game.apply_move(mv).unwrap();
        eprintln!("AFTER OPP MOVE\n{}", self.game);
        mv
    }
}
