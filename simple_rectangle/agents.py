from rect_world import *
import numpy as np
import torch
import torch.nn as nn
import sys
import random
import time

from queue import PriorityQueue


sys.setrecursionlimit(20000)

MAX_ATTEMPTS = 10000

# the uninformative speaker
class S0:

    # takes in a rectangle and a list of past utterances, return a new utterance
    def utter_once(self, rect_params, past_utterances):
        # convert the rectangle to a callable form
        rect = Rect(*rect_params)
        # sample x,y from 0 to MAX_LEN
        x = np.random.randint(0, MAX_LEN)
        y = np.random.randint(0, MAX_LEN)
        # if (x,y) is in past_utterances, then utter again
        if (x,y) in [x[0] for x in past_utterances]:
            return self.utter_once(rect_params, past_utterances)
        # otherwise, return the new utterance, along with the rectangle evaluated on this new utterance
        else:
            return (x,y), rect(x,y)

    # make multiple utterances
    def utter(self, rect, num_utterances):
        # initialize the list of utterances
        utterances = []
        # make num_utterances utterances
        for i in range(num_utterances):
            # append the new utterance to the list
            utterances.append(self.utter_once(rect, utterances))
        # return the list of utterances
        return utterances

    # make the class into a callable form using the above function
    def __call__(self, rect, num_utterances):
        return self.utter(rect, num_utterances)

# the uninformative listener
class L0:

    # takes in a list of utterances, and infer the parameters of the rectangle
    def infer(self, utterances, num_attempts=0):
        # if we've tried too many times, then return None
        if num_attempts > MAX_ATTEMPTS:
            return None, MAX_ATTEMPTS
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

# the learning based listener
class L_NN_F(nn.Module):

    # initialize the listener
    def __init__(self):
        super(L_NN_F, self).__init__()
        # initilize the parameters
        # mapping from a 200 dimensional vector a hidden_size dimensional hidden layer
        hidden_size = 100
        self.fc1 = nn.Linear(200, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # decode out the parameters
        self.T_dec = nn.Linear(hidden_size, MAX_LEN)
        self.B_dec = nn.Linear(hidden_size, MAX_LEN)
        self.L_dec = nn.Linear(hidden_size, MAX_LEN)
        self.R_dec = nn.Linear(hidden_size, MAX_LEN)
        # initialize the optimizer, using SGD
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    # forward pass through the network
    # utterance match is simply a list of list of utterances (yeah it is ugly)
    def forward(self, utterances_batch):
        utterances_tensor_batch = []
        for utterances in utterances_batch:
            # convert utterances to a tensor
            # make a 2x10x10 tensor
            utterances_tensor = torch.zeros(2,10,10)
            for (x,y),b in utterances:
                if b:
                    utterances_tensor[0,x,y] = 1
                else:
                    utterances_tensor[1,x,y] = 1
            # append the tensor to the batch
            utterances_tensor_batch.append(utterances_tensor)
        # convert the batch to a tensor
        utterances_tensor_batch = torch.stack(utterances_tensor_batch)
        # flatten the utterances_tensor tensor
        utterances_tensor_batch = utterances_tensor_batch.view(-1,200)
        # pass the utterances through the network
        # fc1 followed by a relu
        x = self.fc1(utterances_tensor_batch)
        x = nn.functional.relu(x)
        # fc2 followed by a relu
        x = self.fc2(x)
        x = nn.functional.relu(x)
        # decode the parameters, and output the logits
        t = self.T_dec(x)
        b = self.B_dec(x)
        l = self.L_dec(x)
        r = self.R_dec(x)
        # return the logits
        return t, b, l, r        

    # take a batch of utterances and programs and train
    def train(self, utterances_batch, programs_batch):
        # forward pass through the network to get unigrams
        t, b, l, r = self.forward(utterances_batch)
        # convert t_target to a tensor
        t_target = torch.tensor([prog[0] for prog in programs_batch])
        b_target = torch.tensor([prog[1] for prog in programs_batch])
        l_target = torch.tensor([prog[2] for prog in programs_batch])
        r_target = torch.tensor([prog[3] for prog in programs_batch])
        loss_fun = nn.CrossEntropyLoss()
        # compute the loss
        loss = loss_fun(t, t_target) + loss_fun(b, b_target) + loss_fun(l, l_target) + loss_fun(r, r_target)
        # backpropagate the loss
        loss.backward()
        # clip the gradients and update the parameters
        nn.utils.clip_grad_norm_(self.parameters(), 0.25)
        # update the parameters
        self.optimizer.step()
        # return the loss
        return loss.item()

    # takes in a list of utterances, and make the unigram distribution of the parameters of the rectangle
    def generate_unigram(self, utterances_batch):
        # pass the utterances through the network
        t,b,l,r = self.forward(utterances_batch)
        # convert t, b, l, r to probabilities using softmax
        t = nn.functional.softmax(t, dim=1)
        b = nn.functional.softmax(b, dim=1)
        l = nn.functional.softmax(l, dim=1)
        r = nn.functional.softmax(r, dim=1)
        # unsqueeze and convert them to numpy arrays
        t = t.detach().numpy().reshape(-1)
        b = b.detach().numpy().reshape(-1)
        l = l.detach().numpy().reshape(-1)
        r = r.detach().numpy().reshape(-1)
        # return them
        return t,b,l,r

    # at inference time
    # takes in a list of utterances, and infer the parameters of the rectangle
    def infer(self, utterances, num_attempts=0):
        # get the unigram distribution of the rectangle parameters
        t,b,l,r = self.generate_unigram([utterances])
        # enumerate from the unigram distribution
        for num_attempts in range(100):
            T = np.random.choice(range(MAX_LEN), p=t)
            B = np.random.choice(range(MAX_LEN), p=b)
            L = np.random.choice(range(MAX_LEN), p=l)
            R = np.random.choice(range(MAX_LEN), p=r)
            # check if the rectangle is consistent with utterances
            if Rect(T,B,L,R).consistent(utterances):
                # check rectangle is legal
                if T+1 < B and L+1 < R:
                    return (T,B,L,R), num_attempts
        # if we can't find a consistent rectangle, print failure, then return None
        # print("FAILURE")
        return None, num_attempts

    # enumerate from the unigram distribution
    def enumerate(self, utterances, budget):
        # get the unigram distribution of the rectangle parameters
        t,b,l,r = self.generate_unigram([utterances])
        # enumerate from the unigram distribution
        
        # sort and rank each unigram distribution
        t_rank = np.argsort(t)[::-1]
        b_rank = np.argsort(b)[::-1]
        l_rank = np.argsort(l)[::-1]
        r_rank = np.argsort(r)[::-1]
        
        # a helper to translate from queue index to rectangle parameters and probability
        def queue_ids_to_rect_and_logprob(queue_ids):
            # if queue_ids is illegal, return negative infinity
            if any(queue_id < 0 or queue_id >= MAX_LEN for queue_id in queue_ids):
                return None, -np.inf
            rect_params = t_rank[queue_ids[0]], b_rank[queue_ids[1]], l_rank[queue_ids[2]], r_rank[queue_ids[3]]
            log_rect_prob = np.log(t[queue_ids[0]]) + np.log(b[queue_ids[1]]) + np.log(l[queue_ids[2]]) + np.log(r[queue_ids[3]])
            return rect_params, log_rect_prob 

        # make the priority queue
        pqueue = PriorityQueue()
        # add the first rectangle to the queue
        # the first q_id and its log probability
        q_id = (0,0,0,0)
        q_logpr = queue_ids_to_rect_and_logprob(q_id)[1]
        # add this information to the queue
        pqueue.put((-q_logpr, q_id))

        # rectangles to return
        ret = []
        been_in_queue = set([q_id])
        # for up to the budget
        for _ in range(budget):
            # check if the queue is empty
            if pqueue.empty():
                break
            # get the next rectangle from the queue
            q_logpr, q_id = pqueue.get()
            # get the rectangle parameters and probability
            q_rect_params, q_rect_prob = queue_ids_to_rect_and_logprob(q_id)
            # check if the rectangle is consistent with utterances
            
            if Rect(*q_rect_params).consistent(utterances):
                ret.append(q_rect_params)

            # enumerate the next four rectangles
            for q_pos in range(4):
                # get the next queue id
                q_id_next = q_id[:q_pos] + (q_id[q_pos] + 1,) + q_id[q_pos+1:]
                # check if this id has not been in the queue before
                if q_id_next not in been_in_queue:
                    # add this id to been_in_queue now
                    been_in_queue.add(q_id_next)
                    # get the next rectangle parameters and logprobability
                    q_rect_params_next, q_rect_logprob_next = queue_ids_to_rect_and_logprob(q_id_next)
                    if q_rect_params_next is not None:
                        # add the rectangle to the queue
                        pqueue.put((-q_rect_logprob_next, q_id_next))
        # return the rectangles
        # print ("I have seen ", len(seen))
        return ret

    # make the class into a callable form using the above function
    def __call__(self, utterances):
        ret = self.enumerate(utterances, budget=100)
        return ret


# the recursive speaker, does not use learning
class S1:

    # initialize the speaker with a target listener
    def __init__(self, listener):
        self.listener = listener

    # takes in a rectangle and a list of past utterances, return a new utterance
    def utter_once(self, rect_param, past_utterances):
        # convert the rectangle to a callable form
        rect = Rect(*rect_param)
        # enumerate over all possible new coordinates
        old_coords = set([x[0] for x in past_utterances])
        all_coords = []
        for x in range(MAX_LEN):
            for y in range(MAX_LEN):
                # check if the coordinate is new
                if (x,y) not in old_coords:
                    all_coords.append((x,y))
        
        best_coord = None
        best_score = -1
        # for each coordinate, query the listener response
        for coord in all_coords:
            # create a hypothetical utterance using this coordinate
            new_utts = past_utterances + [(coord, rect(*coord))]
            # get the unigram distribution from the listener
            t,b,l,r = self.listener.generate_unigram([new_utts])
            # probability of rectangle
            p_rect = t[rect_param[0]]*b[rect_param[1]]*l[rect_param[2]]*r[rect_param[3]]
            if p_rect > best_score:
                best_score = p_rect
                best_coord = coord
        # return the best utterance
        return best_coord, rect(*best_coord)

    # make multiple utterances
    def utter(self, rect, num_utterances):
        # initialize the list of utterances
        utterances = []
        # make num_utterances utterances
        for i in range(num_utterances):
            # append the new utterance to the list
            utterances.append(self.utter_once(rect, utterances))
        # return the list of utterances
        return utterances

    # make the class into a callable form using the above function
    def __call__(self, rect, num_utterances):
        return self.utter(rect, num_utterances)

# the CEGIS speaker, does not use learning
class S_CE:

    # initialize the speaker with a target listener
    def __init__(self, listener, ce_num = 100):
        self.listener = listener
        self.ce_num = ce_num

    # takes in a rectangle and a list of past utterances, return a new utterance
    def utter_once(self, rect_param, past_utterances):

        # enumerate the listener from past utterances up to budget
        inferred_rect_params_list = self.listener(past_utterances)

        # if the listener fails to find any consistent rectangles, speak no further 
        # the current utterance-prog pair is hard enough
        if len(inferred_rect_params_list) == 0:
            # print (f"[S_CE] failed {past_utterances}")
            return None

        # if the most likely rectangle is the target, speak no further
        if inferred_rect_params_list[0] == rect_param:
            print (f"[S_CE] successful {rect_param} {past_utterances}")
            return None



        # otherwise, try to select a coordinate that filter out the most number of params
        # enumerate over all possible new coordinates
        old_coords = set([x[0] for x in past_utterances])
        all_coords = []
        for x in range(MAX_LEN):
            for y in range(MAX_LEN):
                # check if the coordinate is new
                if (x,y) not in old_coords:
                    all_coords.append((x,y))
        # shufflt the coordinates to a random order
        random.shuffle(all_coords)

        # convert all rectangles to callable form
        rect = Rect(*rect_param)
        inferred_rects = [Rect(*x) for x in inferred_rect_params_list]
        best_uttr = None
        best_score = -1
        # for each coordinate, see if the inferred rectangle is consistent with the target rectangle
        for (x,y) in all_coords:
            # produce an utterance from the current rectangle
            b = rect(*(x,y))
            # check how many rectangles are inconsistent with b on x,y
            num_inconsistent = 0
            for r in inferred_rects:
                if not r(*(x,y)) == b:
                    num_inconsistent += 1
            if best_score < num_inconsistent:
                best_score = num_inconsistent
                best_uttr = (x,y),b
        
        return best_uttr

    # make multiple utterances
    def utter(self, rect, num_utterances):
        # initialize the list of utterances
        utterances = []
        # make num_utterances utterances
        for i in range(num_utterances):
            new_utter = self.utter_once(rect, utterances)
            if new_utter is None:
                break
            # append the new utterance to the list
            utterances.append(new_utter)
        # return the list of utterances
        return utterances

    # make the class into a callable form using the above function
    def __call__(self, rect, num_utterances):
        return self.utter(rect, num_utterances)

if __name__ == '__main__':
    # generate a rectangle
    rect = (1,3,4,9)
    # generate a list of utterances
    utters = S0()(rect, 10)
    # visualize the rectangle and the utterances
    Rect(*rect).draw('tmp/rect_orig.png', utters)
    # attempt to infer the rectangle parameters from the utterances
    inferred_rect_params, num_attempts = L0()(utters)
    # visualize the inferred rectangle and the utterances
    Rect(*inferred_rect_params).draw('tmp/inferred_rect_params.png', utters)

    inferred_rect_params, num_attempts = L_NN_F()(utters)
    print (num_attempts)
    Rect(*inferred_rect_params).draw('tmp/inferred_rect_params_nn.png', utters)