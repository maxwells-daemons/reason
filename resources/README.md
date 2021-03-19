# resources

Various data files used by `reason`.

## ffo
A copy of the [FFO endgame test suite](http://www.radagast.se/othello/ffotest.html) by the French Othello Federation, used
for testing and profiling the endgame solver.
File `ffo/<XX>_<YY>.txt` contains endgame boards of depth XX and YY in a text format.
Text format borrowed from [Jeffrey An's Flippy](https://github.com/jeffreyan11/othello_engine).

## wthor
The [WTHOR games database](https://www.ffothello.org/informatique/la-base-wthor/) by the French Othello Federation, used for
training. Games are stored in the WTHOR database format.

## cassio
Self-play games by Stéphane Nicolet's [Cassio](http://cassio.free.fr/), used for training.
Games are stored in the WTHOR database format.

## logistello
Self-play games by Michael Buro's [Logistello](https://skatgame.net/mburo/log.html), used for training.
Games are stored in an ASCII format.

## checkpoints
Model checkpoints from the current modeel.
Not ready for general use yet.