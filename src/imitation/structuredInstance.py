# A lot of this could be enhanced with the abc module for Abstract Base Classes

class StructuredInstance(object):
    
    def __init__(self):
        self.input = None
        self.output = None


class StructuredInput(object):
    def __init__(self):
        raise NotImplementedError


class StructuredOutput(object):

    # it must return an evalStats object with a loss
    def compareAgainst(self, predicted):
        raise NotImplementedError


class EvalStats(object):
    def __init__(self):
        self.loss = 0 