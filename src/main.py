import numpy as np
from utils import plot_learning_curve
from game import Game
from model import Agent
import os

if __name__ == '__main__':
    env = Game()
    n_actions = 4*18*28 + 1 # 2017 actions
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=n_actions, batch_size=batch_size, n_epochs=n_epochs, alpha=alpha)
    training_state = agent.load_models()
    
    if training_state:
        best_score = training_state['best_score']
        start_episode = training_state['episode'] + 1
        print(f"Resuming from episode {start_episode}, best_score: {best_score}")
    else:
        best_score = -float('inf')
        start_episode = 0
        print("Starting fresh training")

    n_games = 10

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(parent_dir, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    figure_file = os.path.join(models_dir, 'clash royale.png')

    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(state['battlefield'], state['elixir'], state['cards'])
            new_state, reward, done = env.step(action, state)
            n_steps += 1
            score += reward
            agent.remember(state, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
                start_episode += 1
            state = new_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(best_score=best_score, episode=start_episode)

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)