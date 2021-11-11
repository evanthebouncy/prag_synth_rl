from rect_world import *
import numpy as np

# the uninformative speaker
class S0:

    # takes in a rectangle and a list of past utterances, return a new utterance
    def utter(self, rect, past_utterances):
        # convert the rectangle to a callable form
        rect = Rect(*rect)
        # sample x,y from 0 to MAX_LEN
        x = np.random.randint(0, MAX_LEN)
        y = np.random.randint(0, MAX_LEN)
        # if (x,y) is in past_utterances, then utter again
        if (x,y) in past_utterances:
            return self.utter(rect, past_utterances)
        # otherwise, return the new utterance, along with the rectangle evaluated on this new utterance
        else:
            return (x,y), rect(x,y)

    # make the class into a callable form using the above function
    def __call__(self, rect, past_utterances):
        return self.utter(rect, past_utterances)

# the uninformative listener
class L0:

    # takes in a list of utterances, and infer the parameters of the rectangle
    def infer(self, utterances, num_attempts=0):
        # generate a guess for a rectangle
        # generate the T,B,L,R parameters
        T = np.random.randint(0, MAX_LEN)
        B = np.random.randint(0, MAX_LEN)
        L = np.random.randint(0, MAX_LEN)
        R = np.random.randint(0, MAX_LEN)
        # check if the rectangle is consistent with utterances
        if Rect(T,B,L,R).consistent(utterances):
            return (T,B,L,R), num_attempts
        else:
            return self.infer(utterances, num_attempts+1)

    # make the class into a callable form using the above function
    def __call__(self, utterances):
        return self.infer(utterances)

if __name__ == '__main__':
    # generate a rectangle
    rect = (1,3,4,9)
    # generate a list of utterances
    utters = []
    for i in range(4):
        utters.append(S0()(rect, utters))
    # visualize the rectangle and the utterances
    Rect(*rect).draw('tmp/rect_orig.png', utters)
    # attempt to infer the rectangle parameters from the utterances
    rect_inferred, num_attempts = L0()(utters)
    # visualize the inferred rectangle and the utterances
    Rect(*rect_inferred).draw('tmp/rect_inferred.png', utters)