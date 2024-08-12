# Gymnasium Environment Explorer

## Overview
This project provides a graphical user interface (GUI) for exploring registered environments in the Gymnasium library (formerly OpenAI Gym). It allows users to quickly search through available environments and view their specifications.

## Features
- Search functionality to filter Gymnasium environments
- Display of detailed environment specifications
- Interactive GUI built with Tkinter

## Prerequisites
To run this project, you need to have the following installed:
- Python
- Gymnasium
- Tkinter (usually comes pre-installed with new versions of Python)

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
2. Use the search bar to filter environments.
3. Click on an environment in the list to view its specifications.

## How it Works
1. The script uses Gymnasium's registry to get a list of all available environments.
2. It creates a GUI using Tkinter, with a search bar, a listbox for environments, and a text area for specifications.
3. As you type in the search bar, it filters the list of environments in real-time.
4. When you select an environment, it displays the environment's specifications in the text area.

## Potential Extensions
- Add functionality to make selected environments to retrieve more information like action space, observation space, and reward range
- Implement a simple agent (e.g., random action) or user-play function to interact with environments
- Add functionality to rename environments (environment ids, namespaces, etc.) or update the registry

## Learning Outcomes
Through this project, I gained:
- Practical experience with Gymnasium, a key library for reinforcement learning
- Experience in creating graphical interfaces with Tkinter
- Practice in working with Python's object-oriented features