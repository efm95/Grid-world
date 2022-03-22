import numpy as np 
import itertools
from Grid_World import GridWorld

problem = GridWorld()

def transition_prob(s, s_next, a):
    return 1 if problem._next_state(s, a) == s_next else 0


def eval_policy(problem, policy, value, gamma=0.9, theta=0.01):
    p = transition_prob
    r = problem._reward

    while True:
        delta = 0
        for s in problem.states:
            v = value[s]
            value[s] = np.sum(
                [
                    p(s, next_s, policy[s])
                    * (r(s, next_s, policy[s]) + gamma * value[next_s])
                    for next_s in problem.states
                ]
            )
            delta = max(delta, abs(v - value[s]))

        if delta < theta:
            return value


def improve_policy(problem, policy, value, gamma=0.9):
    p = transition_prob
    r = problem._reward

    stable = True
    for s in problem.states:
        actions = problem.actions

        b = policy[s]
        policy[s] = actions[
            np.argmax(
                [
                    np.sum(
                        [
                            p(s, next_s, a) * (r(s, next_s, a) + gamma * value[next_s])
                            for next_s in problem.states
                        ]
                    )
                    for a in actions
                ]
            )
        ]
        if b != policy[s]:
            stable = False

    return stable


def policy_iteration(problem, gamma=0.9, theta=0.01):
    # Initialize a random policy
    policy = np.array([np.random.choice(problem.actions) for s in problem.states])
    print("Initial policy")
    problem.print_policy(policy)
    # Initialize values to zero
    values = np.zeros_like(problem.states, dtype=np.float32)

    # Run policy iteration
    stable = False
    for i in itertools.count():
        print(f"Iteration {i}")
        values = eval_policy(problem, policy, values, gamma, theta)
        problem.print_values(values)
        stable = improve_policy(problem, policy, values, gamma)
        problem.print_policy(policy)
        if stable:
            break

    return policy, values


##### Running #####
problem = GridWorld()
policy_iteration(problem, gamma=0.5)
