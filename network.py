#!/usr/bin/python

from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import random
import math
import csv

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
        nodes = self.graph.graph.nodes()
        if type(targets) is int:
            t = []
            for _ in range(targets):
                x = random.choice(nodes)
                while x in t or x in choices:
                    x = random.choice(nodes)
                t.append(x)
            self.win_con = t
        elif type(targets) is not list:
            self.win_con = [targets]
        else:
            self.win_con = targets
        if type(choices) is not list:
            choices = [choices]
        else:
            choices = choices
        self.threshold = threshold
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
            # DEBUG print('===================')
            # DEBUG print('==               ==')
            # DEBUG print('==   New round   ==')
            # DEBUG print('==               ==')
            # DEBUG print('===================')
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

    def tune(
            self,
            choices,
            targets,
            params='depth',
            n = 60,
            time = 5,
            depth = 5,
            history = [],
            threshold = 0.05
        ):
        p_ratio_list = []
        for i in range(n):
            print(i)
            if params == 'depth':
                depth = random.randint(5,200)
            elif params == 'time':
                time = random.randint(1,60)
            elif params == 'threshold':
                threshold = random.random()
            cdict = self.speech_act(choices,targets,time,depth,history,threshold)
            a = cdict['a']['wins']/cdict['a']['trials']
            na = cdict['na']['wins']/cdict['na']['trials']
            p_win = a * na
            entry = {
                'time': time,
                'depth': depth,
                'threshold': threshold,
                'p_win': p_win,
                'a':a,
                'na':na
            }
            p_ratio_list.append(entry)
        #x = [z[params] for z in p_ratio_list]
        #y = [z['p_win'] for z in p_ratio_list]
        return(p_ratio_list)

    def read_data(self, fname, header = True, **kwargs):
        row_list = []
        with open(fname,'r') as csv_file:
            reader = csv.reader(csv_file,**kwargs)
            if header:
                next(reader)
            for row in reader:
                row_list.append(row)
        self.data = row_list

    def _network_from_data(
            self,
            G = None,
            participant_col=0,
            conditions_dict={
                1:{
                    0:'K',
                    1:'S',
                },
                2:{
                    0:'casual',
                    1:'careful'
                },
                3:{
                    0:'t-deletion',
                    1:'t-flapping',
                    2:'schwa-reduction',
                    3:'minimal-variation',
                    4:'combination'
                }
            },
            node_col=4,
            threshold = 0.05
        ):
        rows = self.data
        nodes = [x[node_col] for x in rows if '(' not in x[node_col]]
        nodes = set(nodes)
        participant_answers = {}
        answers_by_condition = {}
        for col in conditions_dict:
            answers_by_condition[col] = {}
            for cond in conditions_dict[col]:
                #print(col,cond, [int(x[col]) == cond for x in rows])
                answer_list = [x[node_col] for x in rows if int(x[col]) == cond]
                #print(col,cond,answer_list)
                answers_by_condition[col][cond] = answer_list
        for row in rows:
            participant = row[participant_col]
            if participant not in participant_answers:
                participant_answers[participant] = []
            participant_answers[participant].append(row[node_col])
        set_count = float(len(participant_answers))
        weights = {}
        for i in nodes:
            j_runs = 0
            if i not in weights:
                weights[i] = {
                    'count': 0, # how many times i occurs
                    'p': 0.0
                }
            for j in nodes:
                if i == j:
                    continue
                j_runs += 1
                if j not in weights[i]:
                    weights[i][j] = {
                        'count': 0, # how many times i and j co-occur
                        'weight': 1.0
                    }
                p_count = 0
                for participant in participant_answers:
                    p_count += 1
                    if i in participant_answers[participant]:
                        if j_runs <= 1:
                            weights[i]['count']+=1
                        if j in participant_answers[participant]:
                            weights[i][j]['count'] += 1
                if weights[i]['count'] > p_count:
                    raise(ValueError(f'Count for {i} is greater than the number of participants.'))
                i_j_count = float(weights[i][j]['count'])
                i_count = float(weights[i]['count'])
                try:
                    weights[i][j]['weight'] = i_j_count / i_count
                except ZeroDivisionError:
                    weights[i][j]['weight'] = 0.0
                try:
                    weights[i]['p'] = i_count / set_count
                except ZeroDivisionError:
                    weights[i]['p'] = 0.0
        for col in conditions_dict:
            p_id = []
            for cond in conditions_dict[col]:
                p_ids = set([x[participant_col] for x in rows if x[col] == cond])
                i = conditions_dict[col][cond]
                counted_nodes = Counter(answers_by_condition[col][cond])
                c_nodes_sum = 0
                for n in counted_nodes:
                    c_nodes_sum += counted_nodes[n]
                if i not in weights:
                    weights[i] = {
                        'count': len(p_ids),
                        'p': float(len(p_ids))/len(participant_answers)
                    }
                for j in nodes:
                    if j not in counted_nodes:
                        weights[i][j] = {
                            'count': 0,
                            'weight': 0.0
                        }
                    else:
                        weights[i][j] = {
                            'count': counted_nodes[j],
                            'weight': float(counted_nodes[j])/c_nodes_sum
                        }
        if not G:
            G = self.graph.graph
        node_list = [x for x in weights]
        edge_list = []
        for i in weights:
            for j in weights[i]:
                if j in ['count','p']:
                    continue
                w = weights[i][j]['weight']
                edge = (i,j,{'weight':w})
                if w > threshold:
                    print(weights[i][j]['count'],weights[i]['count'],edge)
                    edge_list.append(edge)
        G.add_nodes_from(node_list,trials=0,wins=0)
        G.add_edges_from(edge_list)

    def _simulate(self, choices, history, p, d_count = 0):
        # Choose move from choices
        # if move is a win or loss, back propogate
        # otherwise go deeper
        #
        # Local reference to attributes
        # DEBUG print('=======SIM-START========')
        # DEBUG print('Choices: '+str(choices))
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
        # DEBUG print('Total: '+str(total))
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
                    # DEBUG print(choice,w,t)
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
            # DEBUG print('NS: '+str(not_searched))
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
        # DEBUG print('Choice: '+str(choice))
        # DEBUG print('Neighbors :'+str(neighbors))
        p, result = self._check_win(neighbors,history,p,d_count)
        # DEBUG print(result)
        cdict[choice]['p_history'].append(p)
        #cdict[choice]['p'] = mean(cdict[choice]['p_history'])
        if p > 0.0:
            cdict[choice]['trials'] +=1
            cdict[choice]['wins'] += 1
        else:
            cdict[choice]['trials'] +=1
        self.choice_dict = cdict
        # DEBUG print('========SIM-END==========')
        return(p)

    def _conf_int(self,plays,total):
        num = 2.0 * math.log(total)
        frac = num / plays
        conf = math.sqrt(frac)
        return(conf)

    def _check_win(self,neighbors,history,p,d_count):
        # DEBUG print('=========CHECK-START==========')
        # DEBUG print('C_History: '+str(history))
        # DEBUG print('C_Neighbors: ' + str(neighbors))
        # DEBUG print('d_count: '+ str(d_count))
        G = self.graph.graph
        targets = self.win_con
        depth = self.depth
        threshold = self.threshold
        cdict = self.choice_dict
        n_node = history[-1]
        o_node = history[-2]
        # DEBUG print('o_node: '+str(o_node))
        # DEBUG print('n_node: '+str(n_node))
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
        elif n_node in history[:-1]:
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
    return(x)

def main(targets=3, time = 5, depth = 5, history=[], threshold = 0.05):
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
    MCT = MCTsearch(G)
    cdict = MCT.speech_act(choices,targets,time,depth,history,threshold)
    print(MCT.win)
    return(cdict)

#plt.subplot(111)
#nx.draw_shell(G, with_labels=True, font_weight='bold')
#plt.show()
