from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy import special


# c -> minimum number of correct individuals
# n -> individuals in group
# p -> probability a specific individual is correct
def group_correctness(p: float, n: int, c: int) -> float:
    res = 0

    for i in range((n - c) + 1):
        bc = special.binom(n, i)
        res += bc * (p ** (n - i)) * ((1 - p) ** i)

    return res


def weighted_group_correctness(strong_p: float, strong_weight: float, weak_p: float, weak_n: int) -> float:
    res = 0

    strong_target = int(((weak_n + 1 * strong_weight) / 2) - strong_weight + 1)
    not_strong_target = int(((weak_n + 1 * strong_weight) / 2) + 1)

    res += (1 - strong_p) * group_correctness(weak_p, weak_n, not_strong_target)
    res += strong_p * group_correctness(weak_p, weak_n, strong_target)

    return res


weights = np.arange(0, 30, 1)

plt.plot(weights, [weighted_group_correctness(0.8, w, 0.85, 10) for w in weights], label='$p_{weak}=0.85$')
plt.plot(weights, [weighted_group_correctness(0.8, w, 0.6, 10) for w in weights], label='$p_{weak}=0.6$')
plt.plot(weights, [weighted_group_correctness(0.8, w, 0.4, 10) for w in weights], label='$p_{weak}=0.4$')
plt.title('Weighted probability given different weights for strong classifier')
plt.ylabel('$p_{weighted}$')
plt.xlabel('$w_{strong}$')
plt.legend()
plt.savefig('ex3b.pdf')
plt.show()

# 3.e
model_weight = lambda error: np.log((1 - error) / error)

errors = np.arange(0, 1, 0.01)

plt.plot([model_weight(error) for error in errors], errors)
plt.xlabel('$\\alpha_m$')
plt.ylabel('$eer_m$')
plt.title('AdaBoost model weight function')
plt.savefig('ex3d.pdf')
plt.show()
