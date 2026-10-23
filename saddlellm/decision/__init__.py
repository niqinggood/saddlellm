"""Experimental structured decisions; importing this package loads no model."""
from importlib import import_module

__all__ = ["ChoiceDecisionModel", "DecisionModelConfig", "FrozenTextEncoder", "last_valid_token_pool",
           "TypedDecisionModel", "TypedDecisionConfig", "TypedTrainingConfig", "train_typed", "evaluate_typed"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    modules = {"DecisionModelConfig": ".Schema", "TypedDecisionModel": ".TypedModel",
               "TypedDecisionConfig": ".TypedSchema", "TypedTrainingConfig": ".TypedTraining", "train_typed": ".TypedTraining",
               "evaluate_typed": ".Evaluation"}
    module = modules.get(name, ".Model")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
