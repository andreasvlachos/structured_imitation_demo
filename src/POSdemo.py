import imitation

from sacred import Experiment

ex = Experiment()


# We first need to define the input
class POSInput(imitation.StructuredInput):
    def __init__(self, tokens):
        self.tokens = tokens

    def __str__(self):
        return " ".join(self.tokens)


# Then the NER eval stats
class POSEvalStats(imitation.EvalStats):
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.incorrect = 0
        self.accuracy = 1.0


# Then the output
class POSOutput(imitation.StructuredOutput):
    # the output is tagged entities in the sentence, not just tags
    def __init__(self, tags=None):
        self.tags = []
        if tags!=None:
            self.tags = tags

    def __str__(self):
        return " ".join(self.tags)

    def compareAgainst(self, predicted):
        if len(self.tags) != len(predicted.tags):
            print("ERROR: different number of tags in predicted and gold")

        pos_eval_stats = POSEvalStats()
        for i,pred_tag in enumerate(predicted.tags):
            if pred_tag == self.tags[i]:
                pos_eval_stats.correct += 1
            else:
                pos_eval_stats.incorrect += 1

        pos_eval_stats.accuracy = pos_eval_stats.correct/len(predicted.tags)

        return pos_eval_stats


class POSInstance(imitation.StructuredInstance):
    def __init__(self, tokens, tags=None):
        super().__init__()
        self.input = POSInput(tokens)
        self.output = POSOutput(tags)


class POSTransitionSystem(imitation.TransitionSystem):

    class WordAction(imitation.TransitionSystem.Action):
        def __init__(self):
            super().__init__()

    # the agenda for word prediction is one action per token
    def __init__(self, structured_instance=None):
        super().__init__(structured_instance)
        # Assume 0 indexing for the tokens
        if structured_instance == None:
            return
        #print("agenda init")
        for tokenNo, token in enumerate(structured_instance.input.tokens):
            newAction = self.WordAction()
            newAction.tokenNo = tokenNo
            self.agenda.append(newAction)

    #TODO: make it static?
    def expert_policy(self, structured_instance, action):
        # just return the next action
        return structured_instance.output.tags[action.tokenNo]

    def updateWithAction(self, action, structuredInstance):
        # add it as an action though
        self.actionsTaken.append(action)

    # all the feature engineering goes here
    def extractFeatures(self, structured_instance, action):
        # e.g the word itself that we are tagging
        features = {"currentWord=" + structured_instance.input.tokens[action.tokenNo]: 1}

        # features based on the previous predictionsof this stage are to be accessed via the self.actionsTaken
        # e.g. the previous action
        if len(self.actionsTaken) > 0:
            features["prevPrediction=" + self.actionsTaken[-1].label] = 1
        else:
            features["prevPrediction=NULL"] = 1

        # features based on earlier stages via the state variable.

        return features

    def to_output(self):
        """
        Convert the action sequence in the state to the
        actual prediction, i.e. a sequence of tags
        """
        tags = []
        for action in self.actionsTaken:
            tags.append(action.label)
        return POSOutput(tags)


class POSTagger(imitation.ImitationLearner):
    # specify the transition system
    transitionSystem = POSTransitionSystem

    def __init__(self):
        super().__init__()


@ex.automain
def toy_experiment():
    # load the training data!
    trainingInstances = []

    trainingInstances.extend([POSInstance(["I", "can", "fly"], ["Pronoun", "Modal", "Verb"])])

    trainingInstances.extend([POSInstance(["I", "can", "meat"], ["Pronoun", "Verb", "Noun"])])

    trainingInstances.extend(30*[POSInstance(["I", "can", "fly"], ["Pronoun", "Modal", "Verb"])])

    trainingInstances.extend(10*[POSInstance(["I", "can", "meat"], ["Pronoun", "Verb", "Noun"])])

    # TODO: it would be nice to be able to somehow learn to avoid the original error too
    # TODO: But this would require cost-sensitive learning and LOLS/V-DAgger/SEARN which would be a bit more work first
    tagger = POSTagger()
    #tagger.possibleLabels = set()
    #for tr in trainingInstances:
    #    for tag in tr.output.tags:
    #        tagger.possibleLabels.add(tag)

    print("With standard supervised training")
    # set the params
    params = POSTagger.params()
    # Setting the iterations to one means on iteration, i.e. exact imitation. The learning rate becomes irrelevant then
    params.iterations = 1

    tagger.train(trainingInstances, params)

    print(tagger.labelEncoder.classes_)
    print(tagger.vectorizer.inverse_transform(tagger.model.coef_))

    print(tagger.predict(trainingInstances[0]).to_output())
    print(tagger.predict(trainingInstances[1]).to_output())

    print("With DAgger training")
    # set the params
    paramsImit = POSTagger.params()
    paramsImit.iterations = 2
    paramsImit.learningParam = 1

    tagger.train(trainingInstances, paramsImit)

    print(tagger.labelEncoder.classes_)
    print(tagger.vectorizer.inverse_transform(tagger.model.coef_))

    print(tagger.predict(trainingInstances[0]).to_output())
    print(tagger.predict(trainingInstances[1]).to_output())
