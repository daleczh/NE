import numpy as np

import os

import copy

import time

from docplex.mp.model import Model


def mul(a):
    b = 1

    for i in a:
        b *= i

    return b


class Game:

    def __init__(self, n_agent, n_action, cmd_string):

        self.n_agent = n_agent  # agent的数目

        self.n_action = n_action  # 每个agent的action的数目

        self.result = []  # 找到的nash均衡

        shape = copy.deepcopy(n_action)

        shape.append(n_agent)

        self.reward = np.empty(mul(shape))

        cmd_string = "java -jar gamut.jar " + cmd_string + " -f Game.txt"

        os.system(cmd_string)

        sum = mul(n_action)

        print(sum)

        with open('Game.txt', 'r') as f:

            while 1:

                ss = f.readline()

                if (ss != "" and ss[0] == '['):

                    sum -= 1

                    ss = ss.replace(':', '')

                    ss = ss.replace('[', '')

                    ss = ss.replace(']', '')

                    ss = ss.replace('	', '')

                    ss = ss.split()

                    pos = 0

                    for i in range(n_agent):
                        pos *= n_action[i]

                        pos += int(ss[i]) - 1

                    pos *= n_agent

                    for i in range(n_agent):
                        self.reward[pos + i] = float(ss[n_agent + i])

                if sum == 0: break

            self.reward = self.reward.reshape(shape)  # reward 最终的结构为: [a1][a2]...[an][k]表示n个人的动作分别为a1~an时第k个人的reward

            print(self.reward)


def temp(x):
    xx = abs(x[0] - x[1])
    yy = x[0] + x[1]
    return xx + yy * 1000;


def generate_pair(n_action):
    a = []

    for i in range(n_action[0]):

        for j in range(n_action[1]):
            a.append((i + 1, j + 1))

    a.sort(key=temp)

    return a


def count(x):
    cnt = 0

    while x:
        cnt += x & 1

        x >>= 1

    return cnt


def generate_set(n, m):
    #    print('g_set',n,m)

    b = []

    for i in range(1, n + 1):

        if count(i) == m and (i | n) == n:
            b.append(i)

    return b


def generate_action(s):
    b = []

    i = 0

    while (1 << i) <= s:

        if (1 << i) & s:
            b.append(i)

        i += 1

    return b


def dominated(game, i, a, A, debug=0):
    # return False

    #    print('test',i,a,A, debug)

    l_a = [generate_action(A[0]), generate_action(A[1])]

    for ap in l_a[i]:

        if ap != a:

            cnt = 0

            for aa in l_a[1 - i]:

                if i == 0:

                    if game.reward[ap][aa][i] > game.reward[a][aa][i]:

                        cnt += 1

                    else:

                        break

                else:

                    if game.reward[aa][ap][i] > game.reward[aa][a][i]:

                        cnt += 1

                    else:

                        break

            if cnt == len(l_a[1 - i]): return True

    #    print('succeed')

    return False


def LP(game, s1, s2):
    # which actions are valid?

    valid_action_1 = generate_action(s1)

    valid_action_2 = generate_action(s2)

    # create a model

    model = Model("Nash_Equilibrium")

    # define variables

    ne_value = model.continuous_var_list(2, lb=-model.infinity)

    action_1 = model.continuous_var_list(game.n_action[0], lb=0, ub=1)

    action_2 = model.continuous_var_list(game.n_action[1], lb=0, ub=1)

    # add constraints

    # constraint 1: sum of probs equals to 1

    model.add_constraint(model.sum(action_1) == 1)

    model.add_constraint(model.sum(action_2) == 1)

    # constraint 2: invalid actions have prob of 0

    for i in range(game.n_action[0]):

        if i not in valid_action_1:
            model.add_constraint(action_1[i] == 0)

    for i in range(game.n_action[1]):

        if i not in valid_action_2:
            model.add_constraint(action_2[i] == 0)

    # constraint 3: each player can achieve Nash Equilibrium value

    for i in range(game.n_action[0]):

        if i in valid_action_1:
            model.add_constraint(model.scal_prod(action_2, game.reward[i, :, 0]) == ne_value[0])

    for j in range(game.n_action[1]):

        if j in valid_action_2:
            model.add_constraint(model.scal_prod(action_1, game.reward[:, j, 1]) == ne_value[1])

    # constraint 4: any invalid action is suboptimal

    for i in range(game.n_action[0]):

        if i not in valid_action_1:
            model.add_constraint(model.scal_prod(action_2, game.reward[i, :, 0]) <= ne_value[0])

    for j in range(game.n_action[1]):

        if j not in valid_action_2:
            model.add_constraint(model.scal_prod(action_1, game.reward[:, j, 1]) <= ne_value[1])

    solution = model.solve()

    if solution == None:

        return False

    else:

        game.result.append(solution[ne_value[0]])

        game.result.append(solution[ne_value[1]])

        print('find ', s1, s2)

        return True


def find_nash_equilibrium(game):
    if game.n_agent == 2:

        a = generate_pair(game.n_action)

        print(a)

        for t in a:

            A1 = (1 << game.n_action[0]) - 1

            S1 = generate_set(A1, t[0])

            # print("S1",S1)

            for s in S1:

                A2 = 0

                for i in range(game.n_action[1]):

                    if not dominated(game, 1, i, (s, (1 << game.n_action[1]) - 1), 1):
                        A2 += 1 << i

                flag = False

                l_a_s = generate_action(s)

                for a_s in l_a_s:

                    if dominated(game, 0, a_s, (s, A2), 2):
                        flag = True

                        break

                if flag:
                    continue

                S2 = generate_set(A2, t[1])

                for s2 in S2:

                    flag2 = False

                    l2_a_s = generate_action(s)

                    for a2_s in l2_a_s:

                        if dominated(game, 0, a2_s, (s, s2), 3):
                            flag2 = True

                            break

                    if flag2:
                        continue

                    # print('LP',s,s2)

                    if LP(game, s, s2):
                        return True

        return False


def main():
    start_time = time.time()

    action = [20, 20]

    game = Game(2, action,
                "-g RandomGame -players 2 -normalize -min_payoff 0 -max_payoff 1 -f BoS.game -actions " + str(action[0]) + " " + str(action[1]))

    if find_nash_equilibrium(game):

        print('success!', game.result)

    else:

        print('failed!')

    end_time = time.time()

    print(end_time - start_time)

main()

