# Theory Overview

This package is guided by the theoretical guarantees in the paper, but some constants and details are adapted to practice.

## Problem setting

We consider list-decodable linear regression with contamination rate \(1-\alpha\):

- Inliers: \((x_i, y_i)\) follow \(y_i = \langle x_i, \ell_\star \rangle + \xi_i\),
  with \(x_i\) sub-Gaussian, mean zero, covariance \(\Sigma\), and \(\xi_i\) sub-Gaussian noise.
- Outliers: arbitrary \((x_i, y_i)\) on the remaining samples.
- The inlier fraction \(\alpha \in (0, 1/2]\) is assumed known.

The algorithm outputs a **list** of candidates \(\{\hat\ell^{(k)}\}\), of size \(O(1/\alpha)\), such that at least one is close to \(\ell_\star\) in \(\ell_2\)-norm.

## Key ingredients

- **Lossless expanders** guarantee that many buckets are “lightly contaminated”:
  they contain \(\Theta(\alpha n / B)\) inliers and \(O(1)\) outliers.
- **Robust aggregation** (median-of-means / geometric median) converts lightly contaminated bucket statistics into accurate global moment estimates:
  \[
  \|\hat\Sigma - \Sigma\|_{\text{op}} \lesssim \sigma_x^2 \sqrt{\frac{d + \log(1/\delta)}{\alpha n}},
  \]
  with a similar bound for \(\hat g - \Sigma \ell_\star\).
- **Spectral filtering** iteratively removes buckets that drive up variance along leading eigen-directions of residual covariance, reducing adversarial leverage.
- **Seeding and clustering** ensure that, with enough seeds, at least one candidate is in the basin of attraction of \(\ell_\star\), and the final list size is controlled.

The implementation follows these ideas but:

- Uses an **efficient random expander** construction rather than explicit expanders,
- Uses practical thresholds and stopping criteria,
- Exposes all parameters (`alpha`, `buckets`, `repetitions`, etc.) as estimator hyperparameters.

Formal proofs (with caveats) can be found in the paper.
