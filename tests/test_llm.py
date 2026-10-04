import pytest
import numpy as np
import torch

from sat.llm import _Encoded, _vocab, answer_tokens, record_text, survival_curves


def test_answer_tokens_paper_example():
    cuts = np.array([0, 1, 2, 3, 4, 5.0])
    # no event in 3 units, censored at 3 -> N N N C C   (paper: "NNNCC")
    assert answer_tokens([0, 0], [3.0, 3.0], cuts) == ["N", "N", "N", "C", "C"]
    # event 1 in unit 2, event 2 in unit 4, followed until 5
    assert answer_tokens([1, 1], [1.5, 3.5], cuts) == ["N", "E1", "N", "E2", "C"]
    # two events in the same unit -> combination token
    assert answer_tokens([1, 1], [2.2, 2.8], cuts)[2] == "E1+E2"


def test_record_text_uses_values_only_for_numeric_tokens():
    s = record_text("c_sex_F x_age", [1.0, 0.5], [0, 1])
    assert s == "Patient: c_sex_F, x_age 0.50. Outcome:"


class _Const(torch.nn.Module):
    """Emits fixed next-token logits (identity output layer), so the curves can be
    checked by hand."""

    def __init__(self, logits):
        super().__init__()
        self.l = logits

    @property
    def base_model(self):
        l = self.l

        def body(input_ids, attention_mask):
            b, L = input_ids.shape
            return type("O", (), {"last_hidden_state": l.expand(b, L, -1)})()

        return body

    def get_output_embeddings(self):
        return torch.nn.Identity()


def test_survival_from_token_probabilities():
    vocab = _vocab([], K=2)  # C, E1, E2, N
    special = {s: i for i, s in enumerate(vocab)}
    # p(N)=0.5, p(E1)=0.2, p(E2)=0.1, p(C)=0.2 at every unit
    p = {"N": 0.5, "E1": 0.2, "E2": 0.1, "C": 0.2}
    logits = torch.log(torch.tensor([[p[s] for s in vocab]]))[None]
    enc = _Encoded.__new__(_Encoded)
    enc.items = [([0, 0, 0], [])]
    surv = survival_curves(_Const(logits), enc, special, vocab, K=2, T=3, device="cpu")
    h1, h2 = 0.2 / 0.8, 0.1 / 0.8  # censoring mass removed
    np.testing.assert_allclose(surv[0, 0], [1, 1 - h1, (1 - h1) ** 2, (1 - h1) ** 3], atol=1e-6)
    np.testing.assert_allclose(surv[0, 1], [1, 1 - h2, (1 - h2) ** 2, (1 - h2) ** 3], atol=1e-6)


def test_deephit_loss_by_hand():
    from sat.llm import deephit_loss

    # one subject, K=1, hazards 0.1, 0.2, 0.3 over three units
    haz = torch.tensor([[[0.1], [0.2], [0.3]]])
    # event in unit 2 (0-based 1): -log[(1-0.1) * 0.2]
    ev = deephit_loss(haz, torch.tensor([[1]]), torch.tensor([[1.0]]))
    assert ev.item() == pytest.approx(-np.log(0.9 * 0.2), rel=1e-5)
    # censored in unit 2: -log[(1-0.1)(1-0.2)]
    ce = deephit_loss(haz, torch.tensor([[1]]), torch.tensor([[0.0]]))
    assert ce.item() == pytest.approx(-np.log(0.9 * 0.8), rel=1e-5)


def test_illegal_mass_counts_only_positions_after_c():
    from sat.llm import illegal_mass

    vocab = _vocab([], K=1)  # C, E1, N
    c = vocab.index("C")
    labels = torch.tensor([[vocab.index("N"), c, c]])
    probs = torch.zeros((1, 3, len(vocab)))
    probs[0, :, c] = torch.tensor([0.1, 0.6, 0.9])  # p(C) per position
    # positions after the first [C]: only index 2 -> illegal mass 1 - 0.9
    assert illegal_mass(probs, labels, c).item() == pytest.approx(0.1, rel=1e-6)


def test_event_units_match_answer_tokens():
    from sat.llm import event_units

    cuts = np.array([0, 1, 2, 3, 4, 5.0])
    unit, obs = event_units([1, 0], [1.5, 3.0], cuts)
    assert unit.tolist() == [1, 2] and obs.tolist() == [1.0, 0.0]
    assert answer_tokens([1, 0], [1.5, 3.0], cuts)[1] == "E1"
