"""Run evaluation and action-sequence simulations (supplied).

Tasks: compare per-action accuracy, exact-sequence accuracy and NLL.
Inspect the paths: which predictions hit a wall or the snake's body?
Why can individually plausible actions form an implausible whole sequence?
Do not adjust your network using these held-out test examples.
"""
from supplied_evaluation import main

if __name__ == '__main__':
    main()
