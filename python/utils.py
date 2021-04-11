import pytorch_lightning as pl
import torch
import wandb

from python import game


# From: https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303,
# released under the Apache 2.0 license.
def masked_log_softmax(
    vector: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def soft_crossentropy(predicted_logprobs, target_probs):
    """
    Cross-entropy loss capable of handling soft target probabilities.
    """
    return -(target_probs * predicted_logprobs).sum(1).mean(0)


class VisualizePredictions(pl.Callback):
    """
    Periodically visualize the distribution of model predictions on training data.
    """

    def __init__(self, batch_period: int):
        self._batch_period = batch_period

    def on_train_batch_end(
        self, trainer, module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % self._batch_period:
            return

        outputs = outputs[0][0]["extra"]

        # Visualize model predictions on the first example in the batch
        board = outputs["batch"].board[0, :2]
        board_img = torch.zeros([3, 8, 8], dtype=float, device="cpu")
        board_img[0] = board[0]  # Active player's stones are red
        board_img[2] = board[1]  # Opponent's stones are blue

        # Show legal moves in green
        legal_moves = torch.clone(board_img)
        legal_moves[1] = outputs["batch"].move_mask[0].detach().cpu()

        # Show policy target in green
        policy_target = torch.clone(board_img)
        policy_target[1] = outputs["batch"].policy_target[0]

        # Show policy predictions in green
        policy_scores_flat = outputs["policy_scores_flat"].detach().cpu()
        if module.hparams.mask_invalid_moves:
            move_mask_flat = outputs["batch"].move_mask.flatten(1).detach().cpu()
            policy_scores_flat[~move_mask_flat] = float("-inf")
        policy_probs_flat = policy_scores_flat.softmax(1)

        policy_preds = torch.clone(board_img)
        policy_preds[1] = policy_probs_flat[0].reshape(
            [game.BOARD_EDGE, game.BOARD_EDGE]
        )

        module.logger.experiment.log(
            {
                "policy/distribution": wandb.Histogram(policy_probs_flat),
                "value/distribution": wandb.Histogram(outputs["value"].detach().cpu()),
                "trainer/global_step": trainer.global_step,
                "visualization/legal_moves": wandb.Image(
                    legal_moves, caption="Legal moves"
                ),
                "visualization/policy_preds": wandb.Image(
                    policy_preds, caption="Policy probabilities"
                ),
                "visualization/policy_target": wandb.Image(
                    policy_target, caption="Target probabilities"
                ),
            }
        )
