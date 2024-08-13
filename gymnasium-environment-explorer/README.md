# Gymnasium Environment Explorer

## Overview
This project provides a graphical user interface (GUI) for exploring registered environments in the Gymnasium library (formerly OpenAI Gym). It allows users to quickly search through available environments, view their specifications, examine detailed information about each environment's action space, observation space, and reward range, play supported environments with customizable key mappings, and run random agents.

## Features
- Real-time search functionality to filter Gymnasium environments
- Display of detailed environment specifications, including action space, observation space, and reward range
- Interactive GUI built with Tkinter, featuring a responsive layout
- Scrollbars for improved user experience
- Custom key mapping for playing supported environments
- Support for no-op actions in playable environments
- Random agent functionality with customizable run options

## Prerequisites
To run this project, you need to have the following installed:
- Python
- Gymnasium
- Tkinter (usually comes pre-installed with new versions of Python)
- Some environments may require additional dependencies, such as Box2D, Pygame, or MuJoCo.

## Installation
1. Clone this repository or download the source code.
2. Install the required packages:
   ```
   pip install gymnasium
   ```

## Usage
1. Run the script:
   ```
   python environment_explorer.py
   ```
2. Use the search bar at the top to filter environments in real-time.
3. Click on an environment in the list to view its detailed specifications.
4. Scroll through the environment details in the right panel.
5. To play a supported environment:
   - Select it and click the "Play Selected Environment" button.
   - In the key mapping dialog, customize the controls or use the preset mappings.
6. To run a random agent:
   - Select an environment and click the "Run Random Agent" button.
   - Choose the run mode (continuous, steps, or episodes) and set parameters in the dialog.

## How it Works
1. The script uses Gymnasium's registry to get a list of all available environments.
2. It creates a GUI using Tkinter, with a search bar, a listbox for environments, and a text area for detailed specifications.
3. As you type in the search bar, it filters the list of environments in real-time.
4. When you select an environment, it creates an instance of that environment to retrieve detailed information about its action space, observation space, and reward range.
5. The detailed information is displayed in the text area, along with other specifications from the environment, and the environment is closed.
6. For playable environments, it creates a custom key mapping dialog and uses Gymnasium's play utility to run the environment.
7. For running random agents, it provides options for continuous running, running for a specific number of steps, or running for a specific number of episodes.

## Limitations
- Only works with environments that support 'rgb_array' render mode for play functionality
- Complex action spaces may not be fully supported for play functionality

## Potential Extensions
- Add functionality to rename environments (environment ids, namespaces, etc.) or update the registry
- Add visualization capabilities for supported environments
- Implement functionality to compare multiple environments side by side
- Support for continuous action spaces in play functionality
- Implement more sophisticated AI agents (e.g., Q-learning, policy gradients, baselines)

## Learning Outcomes
Through this project, I gained:
- Practical experience with Gymnasium, a key library for reinforcement learning
- Improved understanding of reinforcement learning environments and their parameters
- Experience in creating responsive and user-friendly graphical interfaces with Tkinter
- Practice in working with Python's object-oriented features and event-driven programming
- Skills in handling and displaying complex, nested data structures
- Experience in implementing custom dialog boxes and user input handling
- Understanding of random agents in reinforcement learning environments
