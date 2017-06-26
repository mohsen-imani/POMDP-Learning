# this function update the model
#def update_model(model, current_state, action, next_state, reward):
#./out/769_15x15_0.01_1_0.7.csv
# check obstacle
import sys, math, random, pickle, numpy

global  q, epsilon, m, n, a, o,with_obstacle, bel, b_tmp, cos, sin,bel_dict, Probability, Q

sin = {}
cos = {}
cos[0] = 1
cos[90] = 0
cos[180] = -1
cos[270] = 0
cos[360] = 1
sin[0] = 0
sin[90] = 1
sin[180] = 0
sin[270] = -1
sin[360] = 0

epsilon = 0.01
bel_dict ={}
#------------------REWARD Function-----------------------
def reward(state):
    if (with_obstacle == 0):
        if ((state[0] < 0 or state[0] >= n) or (state[1] < 0 or state[1] >= m)): # check it out whether we are out of grid or not
            return -100
        elif (state[0] == n-1 and state[1] == m-1):# check it out whether we are in the goal state
            return 100
        else:
            return 0
    else:
        if ((state[0] < 0 or state[0] >= n) or (state[1] < 0 or state[1] >= m)): # check it out whether we are out of grid or not
            return -100
        elif (state[0] == n-1 and state[1] == m-1):# check it out whether we are in the goal state
            return 100
        else: # here we check the obstacles
            #if (((state[0] <= 4 and state[0]>= 3) and (state[1] <= 3 and state[1] >=0 )) or ((state[0] <= 10 and state[0]>= 8) and (state[1] <= (m-1) and state[1] >=((m-1)-2) ))):
            if ((state[0] <= 4 and state[0]>= 3) and (state[1] <= 4 and state[1] >=3 )):
               return -100

            else: # the other state
                return 0


def environment(state, action):
    r=random.random()
    theta = state[2] *90
    theta_radian = theta

    #------------------------------ ACTION FORWARD---------------------------------------
    if (action == 0):# action Forward
        if (r < 0.2):
            return [state, 0]
        else:
            next_state = [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            rwrd = reward(next_state)
            if (rwrd == -100):
                return [state, rwrd]
            else:
                return [next_state, rwrd]
    #------------------------------ ACTION TURN LEFT---------------------------------------
    if (action == 1): # action turn left
        if (r < 0.1): # theta should not be more than 360 degree
            return [state, 0]
        else:
            theta = theta + 90
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                return [state, rwrd]
            else:
                return [next_state, rwrd]
    #------------------------------ ACTION BACKWARD---------------------------------------
    if (action == 2): # action backward
        if (r < 0.2): # theta should not be more than 360 degree
            return [state, 0]

        else:
            theta = theta + 180
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta  # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                return [state, rwrd]
            else:
                return [next_state, rwrd]
    #------------------------------ ACTION TURN RIGHT---------------------------------------
    if (action == 3): # action turn right
        if (r < 0.1): #theta should not be more than 360 degree
            return [state, 0]
        else:
            theta = theta + 270
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                return [state, rwrd]
            else:
                return [next_state, rwrd]

#-------------------------This function takes an action based on epsilon-greedy algorithm------------
def take_action(state):
    r = random.random()
    if (r < epsilon):
        return random.randint(0,a - 1)
    else:
        e= q[state[0]][state[1]][state[2]][:]
        Q_max = e[0]
        ind_max = []
        for i in range(a): # this FOR finds the maximum Q value
            if (Q_max <= e[i]):
                Q_max = e[i]
        for i in range(a): # this part breaks the tie
            if (e[i] == Q_max):
                ind_max.append(i)
        return ind_max[random.randint(0,len(ind_max)-1)]

def p_s_prime_given_s_and_a(s_prime, state, action):
    theta = state[2] *90
    theta_radian = theta 
    for d in range(len(state)):
        if d == 0 and ( state[d] < 0 or s_prime[d] < 0 or  state[d] > (n-1)  or s_prime[d] > (n-1)):
            #print "state", state, " s_prime", s_prime
            return 0
        if d == 1 and (state[d] < 0 or s_prime[d] < 0  or state[d] > (m-1)  or s_prime[d] > (m-1)):
            #print "state", state, " s_prime", s_prime
            return 0
    #------------------------------ ACTION FORWARD---------------------------------------
    if (action == 0):# action Forward
            next_state = [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    #------------------------------ ACTION TURN LEFT---------------------------------------
    if (action == 1): # action turn left
            theta = theta + 90
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    #------------------------------ ACTION BACKWARD---------------------------------------
    if (action == 2): # action backward
            theta = theta + 180
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    #------------------------------ ACTION TURN RIGHT---------------------------------------
    if (action == 3): # action turn right
            theta = theta + 270
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    #print "currnet state",  state
    #print "state prime",  s_prime
    #print "next state",  next_state

    if (next_state == s_prime and next_state != state) and (action == 2 or action == 0):
        p = 0.8
    elif (next_state == s_prime and next_state != state) and (action == 3 or action == 1):
        p = 0.9
    elif (state == s_prime and next_state != state) and (action == 2 or action == 0):
        p = 0.2
    elif (state == s_prime and next_state != state) and (action == 3 or action == 1):
        p = 0.1
    elif (next_state == s_prime and next_state == state):
        p = 1
    else:
        p = 0
    return p


def sigma_transition(s,action):
    sigma = 0
    #print "----------------------Start----------------------------"
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [0,1,2,3]:
                s_i = [s[0]+i, s[1]+j, k]
                p = p_s_prime_given_s_and_a(s, s_i, action)
                if  p != 0.0:
                    sigma = sigma + (p*bel_dict.get((s_i[0],s_i[1],s_i[2]),0))
                    #print "s' = ", s,"  action = ",action, "    s = ", s_i, "   p = ",p,  " b ",bel[s_i[0]][s_i[1]][s_i[2]] ," sigma ", sigma
    #print "----------------------end----------------------------"

    return sigma





def update_belief_s(s, s_prime, action):
    b1 = Probability * sigma_transition(s_prime ,action)
    our_list = get_states(s,s_prime,action)# [(s_prime, action), (s_prime, action), (s_prime, action) ,.....]
    l = len(our_list)
    if l != 0:
        p = (1-Probability)/float(l)
    else:
        b1 = sigma_transition(s_prime ,action)
        p = 0
    lst = []
    for k in range(l):
        b2 = p * sigma_transition(our_list[k],action)
        lst.append(b2)
    #b_=float(b1)/float(sum(lst,b1))
    b_tmp = {}

    global bel_dict
    sum_=float(sum(lst,b1))

    for k in range(l):
        b_tmp[(our_list[k][0],our_list[k][1],our_list[k][2])] = float(lst[k])/sum_
    b_tmp[(s_prime[0],s_prime[1],s_prime[2])] = float(b1)/sum_

    bel_dict = b_tmp
    return


def Q_s_a(action):
    sum_Q = 0
    key = bel_dict.keys()
    values = bel_dict.values()
    for r in range(len(key)):
        sum_Q = sum_Q + values[r]*Q[key[r][0]][key[r][1]][key[r][2]][action]

    return sum_Q



def take_me_next_state(state,action):
    theta = state[2] *90
    theta_radian = theta 
    for d in range(len(state)):
        if d == 0 and ( state[d] < 0 or s_prime[d] < 0 or  state[d] > (n-1)  or s_prime[d] > (n-1)):
            #print "state", state, " s_prime", s_prime
            return 0
        if d == 1 and (state[d] < 0 or s_prime[d] < 0  or state[d] > (m-1)  or s_prime[d] > (m-1)):
            #print "state", state, " s_prime", s_prime
            return 0
    #------------------------------ ACTION FORWARD---------------------------------------
    if (action == 0):# action Forward
            next_state = [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    #------------------------------ ACTION TURN LEFT---------------------------------------
    if (action == 1): # action turn left
            theta = theta + 90
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    #------------------------------ ACTION BACKWARD---------------------------------------
    if (action == 2): # action backward
            theta = theta + 180
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    #------------------------------ ACTION TURN RIGHT---------------------------------------
    if (action == 3): # action turn right
            theta = theta + 270
            if (theta >= 360):
                theta = theta - 360
            theta_radian = theta # theta should not be more than 360 degree
            next_state= [a + b for a, b in zip(state, [cos[theta_radian], sin[theta_radian],0])]
            next_state[2] = int(round(theta/90))
            rwrd = reward(next_state)
            if (rwrd == -100):
                next_state = state
    return next_state
 


def get_states(state,s_prime,action):
    lst =[]
    if state != s_prime:
       lst.append(state)
    for act in range(4):
        s = take_me_next_state(state,act)
        if s not in lst and s != s_prime:
            lst.append(s)
    l = len(lst)
    for k in range(l):
        for t in range(4):
            s = take_me_next_state(lst[k],t)
            if s not in lst and s != s_prime:
                lst.append(s)
    for t in range(4):
        s = take_me_next_state(s_prime,t)
        if s not in lst and s != s_prime:
            lst.append(s)
    #print "s = ",state, "s' = ",s_prime, "action = ", action,  "lst = ", lst
    #print len(lst)
    return lst
        



#initializing
with_obstacle = 0 # our grid world has obstacles
m = 10 # the number of columns
n = 10 # the number of rows
a = 4 # the number of actions
o = 4 # the number of orientation
ALPHA = 0.1 # learning rate
GAMMA = 0.95 # discount factor
Probability = 0.7
# ---------------Initializing the Q matrix (n x m x O x a)-------------
Q = numpy.zeros((n,m,o,a))
q = numpy.zeros((n,m,o,a))
# ------------------------Initial state----------------------------------------------------
init_state = [0,0,0]
episode = 0
step = 0
goal = 0 # this flag will set to one if we are in goal state
#--------------------File setting---------------------------------------------------------

fi= open("project2_data.csv",'w')
#--------------------Initializing belief states-----------------------------------

#-------------------End of Initializing belief states-----------------------------------
current_state = init_state
while (episode < 2000):
    bel_dict = {}
    bel_dict[(0,0,0)] = 1
    episode = episode + 1
    step = 0
    current_state = init_state
    goal = 0
    while (goal == 0):
        action = take_action(current_state)
        out = environment(current_state,action)
        s_prime = out[0]
        rwd = out[1]
        #--------------------------Q(b,a) = sum (b(s) * q(s,a))----------------------------------
        Q_tmp = Q[current_state[0]][current_state[1]][current_state[2]][action] 
        Q[current_state[0]][current_state[1]][current_state[2]][action] = Q_tmp + ALPHA*(rwd + GAMMA*max( Q[s_prime[0]][s_prime[1]][s_prime[2]][:]) -  Q_tmp)
        q[current_state[0]][current_state[1]][current_state[2]][action] = Q_s_a(action)

        update_belief_s(current_state, s_prime, action)

        #fq.write("b(s) = %f" % b[current_state[0]][current_state[1]][current_state[2]],  b[s_prime[0]][s_prime[1]][s_prime[2]])
        #fq.write("  b(s') = %f\n" %b[s_prime[0]][s_prime[1]][s_prime[2]])
        #print "s = ",current_state, " action = ", action, " s' = ",s_prime, " b(s) = ", bel[current_state[0]][current_state[1]][current_state[2]], " b(s') = ", bel[s_prime[0]][s_prime[1]][s_prime[2]]
        current_state = s_prime
        step = step + 1

        if (rwd == 100):
            goal = 1
            fi.write("%d,"%step)
            print (step, "episode", episode)




fi.close()
