from factorgraph import FactorGraph


def LoopyBP(factorgraph: FactorGraph, num_iter: int=1, reset: bool=False):
    """Implements Loopy Belief Propagation with flooding schedule"""
    # Reset states and messages
    if reset: factorgraph.reset()
    # Iterate LBP message passing steps
    for _ in range(num_iter):
        # Send all variable-to-factor messages
        for variable in factorgraph.variables.values():
            variable.send_all_messages()
        # Send all factor-to-variable messages
        for factor in factorgraph.factors.values():
            factor.send_all_messages()
        # Update states
        for variable in factorgraph.variables.values():
            variable.update()

