#!/usr/bin/python

import matplotlib.pyplot as plt
import networkx as nx
import datetime
import random
import math

class IndexicalField():
    def __init__(self,nodes=None,edges=None, threshold=0.05):
        self.graph = nx.Graph()
        self.threshold = threshold
        if nodes:
            self.graph.add_nodes_from(nodes,trials=0,wins=0)
        if edges:
            self.graph.add_nodes_from(edges)

    def _check_state_change(self,state,play):
        raise NotImplementedError

    def possible_associations(self, chain_history, condition='weight'):
        G = self.graph
        pos = []
        c_node = chain_history[-1]
        #o_node = chain_history[-2]
        for neighbor in G.adj[c_node]:
            #candidate = (neighbor,G.adj[c_node][neighbor][condition])
            candidate = neighbor
            pos.append(candidate)
        return(pos)

    def check_success(self,c_hist):
        if len(c_hist) <= 2:
            raise Exception('Chain must contain at least two elements')
        elif c_hist[-1] in c_hist[:-1]:
            return(1)
        elif c_hist[-1] in self.win_con:
            return(2)
        else:
            return(0)

    def add_edges_from(self,edges,**kwargs):
        self.graph.add_edges_from(edges,**kwargs)

    def add_nodes_from(self,nodes,**kwargs):
        self.graph.add_nodes_from(nodes,**kwargs)

class MCTsearch():
    def __init__(self,indexical_field):
        self.graph = indexical_field
        self.history = []
        self.total = 0

    def update(self,chain_history):
        hist = self.history
        h_len = len(hist)
        c_len = len(chain_history)
        if c_len <= h_len:
            print("No new position to add.")
            return()
        elif chain_history[:h_len] != hist:
            raise Exception("Histories do not match.")
        self.history = hist + chain_history[h_len:]

    def speech_act(
            self,
            choices,
            targets,
            time = 30,
            depth=5,
            history=[],
            threshold = 0.05
        ):
        total = 0
        self.depth = depth
        if type(targets) is not list:
            self.win_con = [targets]
        else:
            self.win_con = targets
        if type(choices) is not list:
            choices = [choices]
        else:
            choices = choices
        if threshold:
            self.threshold = threshold
        else:
            self.threshold = self.threshold
        self.choice_dict = {}
        for choice in choices:
            self.choice_dict[choice] = {
                'p': 1.0,
                'UCB': None,
                'p_history': [],
                'wins': 0.0,
                'trials':0.0,
            }
        time_d = datetime.timedelta(seconds=time)
        begin = datetime.datetime.utcnow()
        h_copy = history+['temp']
        self.choices = choices
        while datetime.datetime.utcnow() - begin < time_d:
            print('===================')
            print('==               ==')
            print('==   New round   ==')
            print('==               ==')
            print('===================')
            try:
                choice = random.choice(choices)
            except IndexError:
                raise ValueError("There is nothing to choose from.")
                break
            self.choice_dict[choice]['p'] = 1.0
            _ = h_copy.pop()
            h_copy.append(choice)
            neighbors = self.graph.possible_associations(h_copy)
            p = 1.0
            p = self._simulate(neighbors, h_copy, p)
            self.choice_dict[choice]['trials'] += 1
            if p > 0.0:
                self.choice_dict[choice]['wins'] += 1
            self.choice_dict[choice]['p_history'].append(p)
            total += 1
        return(self.choice_dict)

    def _simulate(self, choices, history, p, d_count = 0):
        # Choose move from choices
        # if move is a win or loss, back propogate
        # otherwise go deeper
        #
        # Local reference to attributes
        print('=======SIM-START========')
        print('Choices: '+str(choices))
        cdict = self.choice_dict
        G = self.graph
        win_con = self.win_con
        depth = self.depth
        depth += 1
        total = 0
        for choice in choices:
            try:
                total += cdict[choice]['trials']
            except KeyError:
                self.choice_dict[choice] = {
                    'p': 1.0,
                    'UCB': None,
                    'p_history': [],
                    'wins': 0.0,
                    'trials':0.0,
                }
        print('Total: '+str(total))
        # Choose association from choices
        if total > 3 * len(choices):
            # if each choice has, on average, 3 choices, always choose the
            #   choice with the highest UCB
            best_choice = [(None,0.0)]
            for choice in choices:
                w = cdict[choice]['wins']
                t = cdict[choice]['trials']
                try:
                    x = w / t
                    c = self._conf_int(t,total)
                    ucb = x + c
                except ZeroDivisionError:
                    print(choice,w,t)
                    ucb = 1.0
                if ucb > 1.0:
                    # If UCB allows for p > 1, ceiling at 1
                    ucb = 1.0
                cdict[choice]['UCB'] = ucb
                if ucb > best_choice[0][1]:
                    best_choice = [(choice,ucb)]
                elif ucb == best_choice[0][1]:
                    # If UCB are identical, either due to chance or
                    #   ceiling affect, add to list to be chosen from at
                    #   random
                    best_choice.append((choice,ucb))
            bc = random.choice(best_choice)
            choice = bc[0]
            p = bc[1]
        elif total < len(choices):
            # Ensure that each choice is searched at least once
            not_searched = [x for x in choices if cdict[x]['trials']==0]
            print('NS: '+str(not_searched))
            choice = random.choice(not_searched)
            try:
                p = cdict[choice]['p']
                cdict[choice]['trials'] += 1
            except KeyError:
                p = 1.0
                self.choice_dict[choice] = {
                    'p': 1.0,
                    'UCB': None,
                    'p_history': [],
                    'wins': 0.0,
                    'trials':1.0,
                }
        else:
            choice = random.choice(choices)
            try:
                p = cdict[choice]['p']
                cdict[choice]['trials'] += 1
            except KeyError:
                p = 1.0
                self.choice_dict[choice] = {
                    'p': 1.0,
                    'UCB': None,
                    'p_history': [],
                    'wins': 0.0,
                    'trials':1.0,
                }
        history.append(choice)
        neighbors = G.possible_associations(history)
        # DEBUG print('History: '+str(history))
        print('Choice: '+str(choice))
        print('Neighbors :'+str(neighbors))
        p, result = self._check_win(neighbors,history,p,d_count)
        print(result)
        cdict[choice]['p_history'].append(p)
        #cdict[choice]['p'] = mean(cdict[choice]['p_history'])
        if p > 0.0:
            cdict[choice]['trials'] +=1
            cdict[choice]['wins'] += 1
        else:
            cdict[choice]['trials'] +=1
        self.choice_dict = cdict
        print('========SIM-END==========')
        return(p)

    def _conf_int(self,plays,total):
        num = 2.0 * math.log(total)
        frac = num / plays
        conf = math.sqrt(frac)
        return(conf)

    def _check_win(self,neighbors,history,p,d_count):
        print('=========CHECK-START==========')
        # DEBUG print('C_History: '+str(history))
        print('C_Neighbors: ' + str(neighbors))
        print('d_count: '+ str(d_count))
        G = self.graph.graph
        targets = self.win_con
        depth = self.depth
        threshold = self.threshold
        cdict = self.choice_dict
        n_node = history[-1]
        o_node = history[-2]
        print('o_node: '+str(o_node))
        print('n_node: '+str(n_node))
        if d_count >= depth:
            # Loss : depth limit
            p = 0.0
            r = 'Depth loss '+str(depth)
            return(p,r)
        elif len(neighbors) == 1 and neighbors[0] == o_node:
            # Loss : deadend
            p = 0.0
            r = 'Deadend'
            return(p,r)
        elif n_node == o_node:
            # Loss : snake
            p = 0.0
            r = 'Snake'
            return(p,r)
        p = p * G.edge[o_node][n_node]['weight']
        if n_node in targets:
            # Win
            r = 'Win'
        elif p < threshold:
            # Loss
            p = 0.0
            r = 'Below threshold'
        else:
            choices = self.graph.possible_associations(history)
            r = ''
            p = self._simulate(choices,history, p, d_count+1)
            if p > 0.0:
                cdict[n_node]['trials'] +=1
                cdict[n_node]['wins'] += 1
            else:
                cdict[n_node]['trials'] +=1
        self.choice_dict = cdict
        return(p,r)

def mean(l):
    s = float(sum(l))
    n = float(len(l))
    x = s / n

def main(depth = 5):
    G = IndexicalField()
    choices = ['a','na']
    G.add_nodes_from([1,2,3,4,5,6,7,8,9,10]+choices,trials=0,wins=0)
    G.add_edges_from([
        ('a',1,{'weight':.5}),
        ('a',3,{'weight':.3}),
        ('a',5,{'weight':.2}),
        ('na',2,{'weight':.7}),
        ('na',4,{'weight':.3}),
        (1,7,{'weight':1.0}),
        (2,6,{'weight':.4}),
        (2,8,{'weight':.2}),
        (2,10,{'weight':.4}),
        (3,6,{'weight':.5}),
        (3,9,{'weight':.5}),
        (4,8,{'weight':1.0}),
        (5,10,{'weight':1.0})
    ])
    target = []
    for _ in range(3):
        x = random.randint(1,10)
        while x in target:
            x = random.randint(1,10)
        target.append(x)
    MCT = MCTsearch(G)
    cdict = MCT.speech_act(choices,target,depth=depth,time=5)
    print(target)
    return(cdict)

#plt.subplot(111)
#nx.draw_shell(G, with_labels=True, font_weight='bold')
#plt.show()
