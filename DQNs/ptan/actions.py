import numpy as np

EPSILON_DEFAULT = 0.05


class ActionSelector:
    # class for sel actions, as the root class
    def __call__(self, scores):
        raise NotImplementedError


class GreedySelector(ActionSelector):
    def __init__(self, epsilon=EPSILON_DEFAULT, selector=None):
        self.epsilon = epsilon
        if selector is None:
            self.selector = ArgmaxSelector()
        else:
            self.selector = selector

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class ProbabilitySelector(ActionSelector):
    # sel action by Probability
    def __call__(self, probability):
        assert isinstance(probability, np.ndarray)
        actions = []
        for index in range(len(probability)):
            prob = probability[index]
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class ArgmaxSelector(ActionSelector):
    # sel action by Argmax
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)