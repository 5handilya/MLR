import random
from ollama import Client
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
###########################################################################################
def generate_random_walk_heatmap(n_iterations=20, n_steps=20, model_name="tinyllama", temperature=1, keyword=""):
    print(f"""generating random walk heatmap for model {model_name}, temperature {temperature}, keyword {keyword}"\n"n_iterations {n_iterations}, n_steps {n_steps}""")
    directions = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (1, 0),
        'down': (-1, 0),
        'stay': (0, 0)
    }
    heatmap = defaultdict(int)
    grid_size = 2*n_steps + 5
    center = 0
    current_pos = (center, center)
    # client = Client()
    client = Client(host="http://192.168.0.15:11434")
    # direction_strings = random.shuffle(['left', 'right', 'up', 'down'])
    # prompt = f"""Always remember the word {keyword}. You are on a random walk inside a {2*grid_size} by {2*grid_size} grid. You can move left or right along the X axis from {-grid_size} to {grid_size}. You can move up and down the Y axis ranging from {grid_size} to {-grid_size}. Your current position is {str(current_pos)}. Where do you want to move? Answer with only ONE word from the following: {direction_strings} to move in that direction."""
    # prompt = "always remember the word {keyword}. respond in one word chosen randomly from the following: [UP, DOWN, RIGHT, LEFT]"
    direction_counts = {
        'left': 0,
        'right': 0,
        'up': 0,
        'down': 0,
        'stay': 0
    }

    for iter in range(n_iterations):
        print(f"iteration {iter}")
        current_pos = (center, center)
        heatmap[current_pos] += 1
        for step in range(n_steps):
            print("step:", step)
            direction_strings = random.shuffle(['left', 'right', 'up', 'down', 'stay'])
            response = client.generate(
                model=model_name,
                prompt = f"""Remember the word "{keyword}". You are on a random walk on a 2-dimensional plane. Your current position (x,y) is {str(current_pos)}. You must pick one direction to move in from the following python list of strings: {direction_strings}. Respond with one word only. Do not use punctuation.""",
                options={
                    'temperature': temperature,
                }
            )
            direction = response.response.strip().lower()
            if direction not in directions:
                direction = 'stay'
            print(response.response, direction)
            direction_counts[direction] += 1
            move = directions[direction]
            current_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            print("dbg response: ", direction, current_pos)
            heatmap[current_pos] += 1
    # print("HEATMAP")
    # print(heatmap) 
    grid = np.zeros((2*grid_size + 1, 2*grid_size + 1))
    for (row, col), count in heatmap.items():
        grid_row = row + grid_size
        grid_col = col + grid_size
        grid[grid_row, grid_col] = count / n_iterations
    
    plt.figure(figsize=(10, 8))
    heatmap_plot = plt.imshow(grid, cmap='viridis', origin="lower", extent=[-grid_size//2 + 1, grid_size//2 + 1, -grid_size//2 + 1, grid_size//2 + 1])
    plt.colorbar(heatmap_plot, label='Visit Probability')
    plt.title(f'Random Walk Heatmap\n'
              f'Model: {model_name}, Temp: {temperature}\n'
              f'Iterations: {n_iterations}, Steps: {n_steps}\n'
              f'Keyword: {keyword}')
    plt.xlabel('X - (left/right)')
    plt.ylabel('Y - (up/down)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.plot(0, 0, 'r+', markersize=5, label='start')
    direction_text = f"Left: {direction_counts['left']}, Right: {direction_counts['right']}, Up: {direction_counts['up']}, Down: {direction_counts['down']}, Stay: {direction_counts['stay']}"
    plt.legend(title=direction_text)

    model_name_safe = model_name.replace('.', 'dot').replace(':', '-').replace(" ", "_")
    temperature_safe = str(temperature).replace('.', 'dot').replace(':', '-').replace(" ", "_")
    keyword_safe = keyword.replace('.', 'dot').replace(':', '-').replace(" ", "_")
    # figname = f"heatmap-{model_name_safe.strip()}-{temperature_safe.strip()}-{keyword_safe.strip()}.jpg"
    figname = f"heatmap-gemma9bsimpo-{temperature_safe}-{keyword_safe}.jpg"
    # figname = f"heatmap-qwen2514-{temperature_safe}-{keyword_safe}.jpg"
    plt.savefig(figname, format='jpg')
    # plt.show()
###########################################################################################
keywords = ["", "order", "chaos", "perfect", "imperfect", "straight up", "curvy", "x", "positive x", "negative x", "y", "positive y", "negative y", "left", "right", "down", "up", "not up", "not down", "north", "south", "east", "west", "fast", "slow", "center", "left edge", "right edge", "hot", "cold"]


for keyword in keywords:
    generate_random_walk_heatmap(
        n_iterations=50,
        n_steps=50,
        # model_name='dimweb/gemma2-9b-it-simpo:Q4_0',
        model_name='gemma2:2b',
        # model_name='llama3.2:3b',
        # model_name='hermes3:3b',
        # model_name='qwen2.5:7b',
        temperature=1,
        keyword=keyword
    )
