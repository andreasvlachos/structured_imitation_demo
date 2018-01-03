from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import random

class ImitationLearner(object):

    # initialize the classifier to be learned
    def __init__(self):
        # Any classifier could be used here
        self.model = LogisticRegression()
        self.vectorizer = DictVectorizer()
        self.labelEncoder = LabelEncoder()

    # this function predicts an instance given the state
    # state keeps track the various actions taken
    # it does not change the instance in any way,
    # it does change the state
    # the predicted structured output is returned in the end
    def predict(self, structured_instance, state=None, expert_policy_prob=0.0):
        if state==None:
            state = self.transitionSystem(structured_instance=structured_instance)

        # predict all remaining actions
        # if we do not have any actions we are done
        while len(state.agenda) > 0:
            # for each action
            # pop it from the queue
            current_action = state.agenda.popleft()
            # extract features and add them to the action
            # (even for the optimal policy, it doesn't need the features but they are needed later on)
            current_action.features = state.extractFeatures(structured_instance=structured_instance, action=current_action)
            # the first condition is to avoid un-necessary calls to random which give me reproducibility headaches
            if (expert_policy_prob == 1.0) or (expert_policy_prob > 0.0 and random.random() < expert_policy_prob):
                current_action.label = state.expert_policy(structured_instance, current_action)
            else:
                # predict (probably makes sense to parallelize across instances)
                # vectorize the features:
                vectorized_features = self.vectorizer.transform(current_action.features)
                # predict using the model
                normalized_label = self.model.predict(vectorized_features)
                # get the actual label (returns an array, get the first and only element)
                current_action.label = self.labelEncoder.inverse_transform(normalized_label)[0]
            # add the action to the state making any necessary updates
            state.updateWithAction(current_action, structured_instance)

        # OK return the final state reached
        return state

    class params(object):
        def __init__(self):
            self.learningParam = 0.1
            self.iterations = 40

    #@profile
    def train(self, structuredInstances, params):
        # create the dataset
        trainingFeatures = []
        trainingLabels = []

        # for each iteration
        for iteration in range(params.iterations):
            # set the expert policy prob
            expertPolicyProb = pow(1-params.learningParam, iteration)
            print("Iteration:"+ str(iteration) + ", expert policy prob:"+ str(expertPolicyProb))
            
            for structuredInstance in structuredInstances:

                # so we obtain the predicted output and the actions taken are in state
                # this prediction uses the gold standard since we need this info for the expert policy actions
                final_state = self.predict(structuredInstance, expert_policy_prob=expertPolicyProb)

                # initialize a second state to avoid having to roll-back
                stateCopy = self.transitionSystem(structured_instance=structuredInstance)
                # The agenda seems to initialized fine
                for action in final_state.actionsTaken:
                    # DAgger just ask the expert
                    stateCopy.agenda.popleft()
                    expert_action_label = stateCopy.expert_policy(structuredInstance, action)

                    # add the labeled features to the training data
                    trainingFeatures.append(action.features)
                    trainingLabels.append(expert_action_label)

                    # take the original action chosen to proceed
                    stateCopy.updateWithAction(action, structuredInstance)

            # OK, let's save the training data and learn some classifiers            
            # vectorize the training data collected
            training_data = self.vectorizer.fit_transform(trainingFeatures)
            # encode the labels
            encoded_labels = self.labelEncoder.fit_transform(trainingLabels)
            # train
            self.model.fit(training_data,encoded_labels)
