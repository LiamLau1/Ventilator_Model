# Ventilator Model for Predicting Respiratory Pressure
Side project to understand and model how to control a mechanical ventilator to maintain pressure in a patient's lungs. Based on a joint Google Brain and Princeton Kaggle competition: https://www.kaggle.com/competitions/ventilator-pressure-prediction/overview

## The Problem
A ventilator pushes air into the lungs using two control variables: a continuous variable u_in ∈ [0, 100], which sets how open the intake valve is, and a binary variable u_out, which decides whether the exhalation valve is open or closed. The lungs are described by a resistance R, how difficult it is to push air through the airways, and a compliance C, how easy the lungs themselves are to inflate. The goal is to predict the airway pressure during the inhalation phase (u_out = 0), scored by the mean absolute error.

The thing I didn't expect: for most breaths u_in isn't set freely. A PI controller reads the pressure and sets u_in from how far the pressure is from a target. So u_in is partly a reaction to the pressure, not only its cause. I think this is why the pressure wiggles seemed to lead the u_in wiggles when I first looked at the data.

## Plan
I don't want to just beat the leaderboard. I want a model that does relatively well, that I can interpret physically, and that is causal, i.e. it only uses what a real ventilator would know at that moment (no bidirectional LSTMs, no pressures that depend on u_in at the end of the breath). It should also be much cheaper to run than the LSTM based models.

1. **Baseline LSTM.** Run the same LSTM with more and more information: only past u_in, past and current u_in, and the whole breath. Do each with and without the starting pressure. This tells me how much of the skill comes from information a physical model could never have.
2. **LSTM + CNN + Transformer.** A scaled down version of what the top Kaggle solutions did, plus a causal version of it.
3. **Inverting the PI controller.** Recover the controller's target and gains from u_in alone and see how far that gets you.
4. **Physics model.** Start from the RC circuit analogy, P(t) = P0 + R·flow(t) + volume(t)/C, and add what it misses one piece at a time:
   - a valve curve (how u_in maps to flow, which saturates when the valve is wide open)
   - a lag between the valve command and the flow
   - extra resistance and compliance from the ventilator and its tubing, not just the lungs
   - a small neural network to soak up whatever is left over

   For each piece I'll check how much it actually improves the score.
5. **Compare everything.** Accuracy vs causality vs interpretability vs cost.

## Repo Layout
- `src/vent/`: data loading, features, metrics, PI controller labelling, training loop, cost measurement, models
- `scripts/`: numbered steps to reproduce everything
- `reports/`: one HTML report per model family (open in a browser)
- `results/`: metrics, training logs and figures (model weights are not committed)
- `tests/`

Put the Kaggle csv files in `data/` and run the scripts in order.

## Status
- [x] Data pipeline, two evaluation tracks, metrics
- [x] Baseline LSTM (5 of 6 runs), `reports/report1_lstm.html`
- [ ] LSTM + CNN + Transformer
- [ ] PI controller inversion
- [ ] Physics model
- [ ] Comparison
