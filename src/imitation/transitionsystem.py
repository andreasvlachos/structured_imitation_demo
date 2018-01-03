# This is a basic definition of the state. But it can be overriden/made more complicated to support:
# - bookkeeping on top of the actions to facilitate feature extraction, e.g. how many times a tag has been used
# - non-trivial conversion of the state to the final prediction


from collections import deque
from copy import deepcopy


class TransitionSystem(object):

    def __init__(self, structured_instance=None):
        # construct action agenda
        self.agenda = deque([])
        self.actionsTaken = []

    class Action(object):
        def __init__(self):
            self.label = None
            self.features = []

        def __deepcopy__(self, memo):
            newone = type(self)()
            newone.__dict__.update(self.__dict__)
            newone.features = deepcopy(self.features)
            return newone

    # extract features for current action in the agenda
    def extractFeatures(self, structured_instance, action):
        pass

    def expert_policy(self, structured_instance, action):
        pass

    def updateWithAction(self, action, structured_instance):
        pass

    # by default it is the same is the evaluation for each stage
    # the object returned by the predict above should have the appropriate function
    @staticmethod
    def evaluate(prediction, gold):
        # order in calling this matters
        return gold.compareAgainst(prediction)


    def to_output(self):
        pass
