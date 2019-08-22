# lookahead_paddle

paddle implement of Lookahead Optimizer

# usage

```python
    sgd = fluid.optimizer.Momentum(
	learning_rate=learning_rate, momentum=0.9, regularization=regularization)
    optimizer = lookahead.LookaheadOptimizer(sgd, alpha=alpha, k=k)
```
