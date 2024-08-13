# Gymnasium Environment Explorer

## Overview
This project provides a graphical user interface (GUI) for exploring registered environments in the Gymnasium library (formerly OpenAI Gym). It allows users to quickly search through available environments, view their specifications, and examine detailed information about each environment's action space, observation space, and reward range.

## Features
- Real-time search functionality to filter Gymnasium environments
- Display of detailed environment specifications, including action space, observation space, and reward range
- Interactive GUI built with Tkinter, featuring a responsive layout
- Scrollbars for improved user experience

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

## How it Works
1. The script uses Gymnasium's registry to get a list of all available environments.
2. It creates a GUI using Tkinter, with a search bar, a listbox for environments, and a text area for detailed specifications.
3. As you type in the search bar, it filters the list of environments in real-time.
4. When you select an environment, it creates an instance of that environment to retrieve detailed information about its action space, observation space, and reward range.
5. The detailed information is displayed in the text area, along with other specifications from the environment.
6. The environment is closed.

## Potential Extensions
- Implement a simple agent (e.g., random action) or user-play function to interact with environments
- Add functionality to rename environments (environment ids, namespaces, etc.) or update the registry
- Add visualization capabilities for supported environments
- Implement functionality to compare multiple environments side by side

## Learning Outcomes
Through this project, I gained:
- Practical experience with Gymnasium, a key library for reinforcement learning
- Improved understanding of reinforcement learning environments and their parameters
- Experience in creating responsive and user-friendly graphical interfaces with Tkinter
- Practice in working with Python's object-oriented features and event-driven programming
- Skills in handling and displaying complex, nested data structures