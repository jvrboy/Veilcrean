"""Reinforcement learning demo: the brain learns to escape a grid maze.

Run:  python examples/demo_learning.py
"""

from neurosense import Brain

GRID = 5           # 5x5 grid; start (0,0), goal (4,4)
ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def step(state, action):
    r, c = state
    dr, dc = MOVES[action]
    nr, nc = max(0, min(GRID - 1, r + dr)), max(0, min(GRID - 1, c + dc))
    new_state = (nr, nc)
    if new_state == (GRID - 1, GRID - 1):
        return new_state, 10.0, True     # reached goal
    return new_state, -0.1, False        # small cost per move


def main():
    brain = Brain(name="maze-runner")
    agent = brain.get_agent("maze", ACTIONS)

    for episode in range(300):
        state, done, moves = (0, 0), False, 0
        while not done and moves < 100:
            action = agent.choose(state)
            next_state, reward, done = step(state, action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            moves += 1

    # Show the learned greedy path
    state, path, done = (0, 0), [(0, 0)], False
    while not done and len(path) < 30:
        action = agent.best_action(state)
        state, _, done = step(state, action)
        path.append(state)
    print(f"Learned path in {len(path) - 1} moves: {path}")
    print(f"Exploration rate decayed to {agent.epsilon:.3f} "
          f"after {agent.total_updates} updates.")

    brain.learn_fact("maze", "is_a", "solvable problem")
    print(brain.think())


if __name__ == "__main__":
    main()
