Subject: Re: CoG discussion — teamfight prediction and exchange-value labels

Hi [Name],

Thank you for your thoughtful message, for sharing the paper, and for your interest in collaborating. It was a pleasure meeting you at CoG, and I would be very happy to explore working together.

Your question about the exchange-value coefficients addresses an issue we have been revisiting for the journal extension. In the version presented at CoG, we assigned fixed, manually specified values and importance weights to events such as kills, shutdowns, and objective acquisitions. The intention was to capture exchanges that a simple kill difference would miss. These were domain-informed design choices rather than coefficients learned from data. Although the prediction target was binary, it was derived from this richer exchange score. I agree that the coefficient choices needed stronger empirical justification.

Since then, we have moved toward a different valuation approach. We train a separate model to estimate a team's eventual match-win probability from the game state available at a given time. We then measure the change in that estimated probability between the state immediately before an engagement and a defined endpoint after it. Our engagement predictor uses only pre-engagement information to predict which team will gain in estimated win probability. We retain the continuous change for analysis, although the current engagement predictor still uses its direction as a binary target.

This lets the valuation model learn how economic and objective information relates to match outcomes, rather than assigning fixed exchange coefficients ourselves. We use out-of-fold valuation predictions for training-match labels, and constrain the endpoint to reduce overlap with subsequent combat. We interpret the result as an operational measure of strategic change, not an isolated causal effect of the fight.

We have run the revised pipeline on a designated corpus of 210,000 matches using patch-based train/validation/test splits, with additional external patch and region evaluations. In a recent controlled comparison, removing explicit objective features worsened match-win probability prediction and changed approximately 5.4% of teamfight labels on the main test patch. These are preliminary findings, but they support retaining objective information in the valuation model.

Your suggestion about more even game states is particularly interesting. We have now drafted a controlled comparison of linear, gradient-boosted tree, and residual neural-network engagement predictors using the same pre-engagement inputs and a prespecified tuning budget. Alongside the full benchmark, we plan to evaluate fights whose pre-engagement match-win estimate is between 40% and 60%, with 45–55% as a secondary sensitivity analysis. These ranges are our proposed operational choices. We will compare against a baseline using only the pre-engagement match-win estimate, and examine calibration and SHAP explanations in the balanced subset. An even overall game state and uncertainty about a particular fight will be treated separately. These new comparisons are planned, not yet executed.

The Kim et al. paper is especially relevant now that probability estimates underpin our labels. We are evaluating calibration, but have not yet implemented its uncertainty-aware loss. Comparing that approach with simpler calibration methods, particularly in balanced game states, seems like a useful direction for collaboration.

Would you be interested in a short call to discuss how we might define and evaluate these balanced or difficult engagement scenarios? I would be happy to share a concise summary of the revised framework and preliminary results beforehand.

Best regards,
Seongeun
