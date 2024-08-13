import gymnasium
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from tkinter.ttk import *

# Function to update the listbox based on the search term
def search_listbox(*args):
    search_term = search_entry.get()
    listbox.delete(0, tk.END)  # Clear the current listbox
    # Populate the listbox with environments that match the search term
    [listbox.insert(tk.END, env_id) for env_id in gymnasium.envs.registration.registry if search_term in env_id]

# Function to display the specifications of the selected environment
def get_env_spec(event):
    selection = listbox.curselection()
    if selection:
        selected_env = listbox.get(selection)  # Get the selected environment
        env_spec = gymnasium.spec(selected_env)  # Get the environment specifications
        env = gymnasium.make(selected_env)  # Create the environment to get more detailed information
        env_spec_text.delete(1.0, tk.END)  # Clear the current text
        env_spec_text.insert(tk.END, f"Environment: {selected_env}\n\n")
        env_spec_text.insert(tk.END, f"Action Space: {env.action_space}\n")
        env_spec_text.insert(tk.END, f"Observation Space: {env.observation_space}\n")
        env_spec_text.insert(tk.END, f"Reward Range: {env.reward_range}\n\n")
        env_spec_text.insert(tk.END, "Other Specifications:\n")
        # Display all other specifications in the text widget
        [env_spec_text.insert(tk.END, f"{key}: {value}\n") for key, value in env_spec.__dict__.items()]
        
        env.close()  # Close the environment to free up resources

# Dialog for setting up key mappings for playing environments
class KeyMappingDialog(simpledialog.Dialog):
    def __init__(self, parent, title, action_space):
        self.action_space = action_space
        self.mappings = {}
        self.noop_entry = None
        super().__init__(parent, title)

    def body(self, frame):
        # Create UI elements for no-op action and key mappings
        # No-op action entry
        ttk.Label(frame, text="No-op (no action) [action index]:").grid(row=0, column=0)
        self.noop_entry = ttk.Entry(frame, width=5)
        self.noop_entry.grid(row=0, column=1)
        self.noop_entry.insert(0, "0")  # Default no-op action

        # Label for entering key for each action
        ttk.Label(frame, text="Enter key for each action:").grid(row=1, column=0, columnspan=2)
        
        # Preset key controls for the first 18 actions
        key_presets = [
            "q", "w", "e", "r", "t", "y", "a", "s", "d", "f", "g", "h", "z", "x", "c", "v", "b", "n"
        ]
        
        # Create a label and entry for each action
        for i in range(self.action_space.n):
            ttk.Label(frame, text=f"Action {i}:").grid(row=i+2, column=0)
            entry = ttk.Entry(frame, width=5)
            entry.grid(row=i+2, column=1)
            # Insert the preset key for the action if it exists
            if i < len(key_presets):
                entry.insert(0, key_presets[i])
            self.mappings[i] = entry

    def apply(self):
        # Get the key mappings and no-op action from the entries
        self.result = {i: entry.get() for i, entry in self.mappings.items() if entry.get()}
        self.noop_action = int(self.noop_entry.get())

# Function to get key mapping for a specific environment
def get_key_mapping(env_id):
    # Create environment, get action space, and close immediately
    env = gymnasium.make(env_id)
    action_space = env.action_space
    env.close()
    
    # Handle key mapping for discrete action spaces
    if isinstance(action_space, gymnasium.spaces.Discrete):
        # Create a dialog for key mapping for discrete action spaces
        dialog = KeyMappingDialog(window, "Key Mapping", action_space)
        if dialog.result:
            # Convert the key mappings to a dictionary
            key_to_action = {v: k for k, v in dialog.result.items()}
            return key_to_action, dialog.noop_action
        
    # Return None for non-discrete action spaces
    return None, None

# Function to play the selected environment with key mappings and no-op action
def play_environment():
    # Get the selected environment from the listbox
    selection = listbox.curselection()
    if selection:
        selected_env = listbox.get(selection[0])
        try:
            # PlayableGameWrapper works only with rgb_array or rgb_array_list render modes
            env = gymnasium.make(selected_env, render_mode="rgb_array")
            # Get the key mapping and no-op action for the environment
            key_mapping, noop_action = get_key_mapping(selected_env)
            if key_mapping is not None:
                # If a key mapping is available, play the environment
                play(env, keys_to_action=key_mapping, noop=noop_action)
            else:
                # If no key mapping is available, show a warning
                messagebox.showwarning("No Key Mapping", "Unable to play due to complex action space.")
        except Exception as e:
            # If an error occurs during environment creation or play, show an error message
            messagebox.showerror("Error", f"Unable to play environment: {str(e)}")
        finally:
            # Ensure the environment is closed, even if an exception occurred
            if 'env' in locals():
                env.close()
    else:
        # If no environment is selected, show a warning
        messagebox.showwarning("No Environment Selected", "Please select an environment from the list.")

# Function to run a random agent in the environment
def run_random_agent(env, mode, steps, episodes, render):
    total_reward = 0
    episode_count = 0
    step_count = 0
    
    while True:
        observation, info = env.reset()
        episode_reward = 0
        
        while True:
            # Render if option is selected
            if render:
                env.render()

            # Take a random action
            action = env.action_space.sample()

            # Perform the action and get the result
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Check if the episode has ended
            if terminated or truncated:
                break

            # Check if we've reached the step limit (if in step mode)
            if mode == "steps" and step_count >= int(steps):
                env.close()
                messagebox.showinfo("Random Agent Results", f"Total reward over {step_count} steps: {total_reward}")
                return

        # Update total reward and episode count
        total_reward += episode_reward
        episode_count += 1

        # Check if we've reached the episode limit (if in episode mode)
        if mode == "episodes" and episode_count >= int(episodes):
            env.close()
            messagebox.showinfo("Random Agent Results", f"Total reward over {episode_count} episodes: {total_reward}")
            return

        # For continuous mode, ask if the user wants to continue after each episode
        if mode == "continuous":
            should_continue = messagebox.askyesno("Continue?", f"Episode {episode_count} finished. Total reward so far: {total_reward}. Continue?")
            if not should_continue:
                env.close()
                messagebox.showinfo("Random Agent Results", f"Total reward over {episode_count} episodes: {total_reward}")
                return

# Function to set up and run the random agent for the selected environment
def run_random_agent_for_environment():
    selection = listbox.curselection()
    if selection:
        selected_env = listbox.get(selection)
        dialog = RandomAgentDialog(window, "Random Agent Settings")
        if dialog.result:
            mode, steps, episodes, render = dialog.result
            try:
                # Create environment with appropriate render mode
                env = gymnasium.make(selected_env, render_mode="human" if render else None)
                run_random_agent(env, mode, steps, episodes, render)
            except Exception as e:
                messagebox.showerror("Error", f"Unable to run random agent: {str(e)}")
            finally:
                if 'env' in locals():
                    env.close()
    else:
        messagebox.showwarning("No Environment Selected", "Please select an environment from the list.")

# Dialog for setting up random agent parameters
class RandomAgentDialog(simpledialog.Dialog):
    def __init__(self, parent, title):
        # Initialize variables for agent settings
        self.mode = tk.StringVar(value="continuous")
        self.steps = tk.StringVar(value="1000")
        self.episodes = tk.StringVar(value="10")
        self.render = tk.BooleanVar(value=True)
        super().__init__(parent, title)

    def body(self, frame):
        # Create UI elements for setting agent parameters
        # Create UI radio buttons for selecting run mode
        ttk.Radiobutton(frame, text="Run continuously", variable=self.mode, value="continuous").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Run for specific steps", variable=self.mode, value="steps").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Run for specific episodes", variable=self.mode, value="episodes").grid(row=2, column=0, sticky="w")
        
        # Create UI label and entry for setting the number of steps
        ttk.Label(frame, text="Number of steps:").grid(row=1, column=1)
        ttk.Entry(frame, textvariable=self.steps, width=10).grid(row=1, column=2)
        
        # Create UI label and entry for setting the number of episodes
        ttk.Label(frame, text="Number of episodes:").grid(row=2, column=1)
        ttk.Entry(frame, textvariable=self.episodes, width=10).grid(row=2, column=2)

        # Create UI checkbox for optionally rendering the environment
        ttk.Checkbutton(frame, text="Render environment", variable=self.render).grid(row=3, column=0, columnspan=3, sticky="w")

    def apply(self):
        # Collect the settings when the user clicks OK
        self.result = (self.mode.get(), self.steps.get(), self.episodes.get(), self.render.get())

# Function to visualize the observation space of the selected environment
def visualize_environment():
    selection = listbox.curselection()
    if selection:
        selected_env = listbox.get(selection)
        try:
            # Create the environment
            env = gymnasium.make(selected_env)

            # Ask user to choose between sample() and reset()
            choice = simpledialog.askstring("Observation Source",
                                            "Choose observation source:\n1. Random sample from space\n2. Initialized environment",
                                            initialvalue="1")
            
            # Get the observation based on user choice
            if choice == "1":
                observation = env.observation_space.sample()
            elif choice == "2":
                observation, _ = env.reset()
            else:
                messagebox.showwarning("Invalid Choice", "Invalid choice.")
                return

            env.close()
            
            # Create a new window for visualization
            viz_window = tk.Toplevel(window)
            viz_window.title(f"Visualization: {selected_env}")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Function to visualize different types of spaces
            def visualize_space(space, obs, ax, title=''):
                # Visualize Box space
                if isinstance(space, gymnasium.spaces.Box):
                    if len(obs.shape) == 1:  # 1D observation
                        ax.bar(range(len(obs)), obs)
                        ax.set_xlabel('Observation Index')
                        ax.set_ylabel('Value')
                    elif len(obs.shape) == 2:  # 2D observation (image-like)
                        ax.imshow(obs, cmap='viridis')
                        ax.axis('off')
                    elif len(obs.shape) == 3:  # 3D observation (RGB image)
                        ax.imshow(obs)
                        ax.axis('off')
                    else:  # Higher dimensional Box
                        ax.text(0.5, 0.5, f"Box Space\nShape: {obs.shape}\nLow: {space.low}\nHigh: {space.high}", 
                                ha='center', va='center')
                        ax.axis('off')
                
                # Visualize Discrete space
                elif isinstance(space, gymnasium.spaces.Discrete):
                    ax.bar(['Observation'], [obs])
                    ax.set_ylabel('Value')
                    ax.set_title(f"{title}\nDiscrete Space\nn: {space.n}, start: {space.start}")
                
                # Visualize MultiBinary space
                elif isinstance(space, gymnasium.spaces.MultiBinary):
                    ax.imshow(obs.reshape(-1, 1), cmap='binary', aspect='auto')
                    ax.set_title(f"{title}\nMultiBinary Space\nShape: {space.shape}")
                    ax.set_xlabel('Value (Black: 0, White: 1)')
                    ax.set_yticks(range(len(obs)))
                    ax.set_yticklabels(range(len(obs)))
                
                # Visualize MultiDiscrete space
                elif isinstance(space, gymnasium.spaces.MultiDiscrete):
                    ax.bar(range(len(obs)), obs)
                    ax.set_xlabel('Dimension')
                    ax.set_ylabel('Value')
                    ax.set_title(f"{title}\nMultiDiscrete Space\nnvec: {space.nvec}")
                    ax.set_xticks(range(len(obs)))
                
                # Visualize Text space
                elif isinstance(space, gymnasium.spaces.Text):
                    ax.text(0.5, 0.5, f"{title}\nText Space\nObservation: '{obs}'\nCharset: {space.charset}\nLength: {len(obs)}/{space.max_length}", 
                            ha='center', va='center')
                    ax.axis('off')
                
                # Visualize Dict space
                elif isinstance(space, gymnasium.spaces.Dict):
                    ax.text(0.5, 0.5, f"{title}\nDict Space\nKeys: {list(space.spaces.keys())}\nObservation: {obs}", 
                            ha='center', va='center')
                    ax.axis('off')
                
                # Visualize Tuple space
                elif isinstance(space, gymnasium.spaces.Tuple):
                    ax.text(0.5, 0.5, f"{title}\nTuple Space\nLength: {len(space.spaces)}\nObservation: {obs}", 
                            ha='center', va='center')
                    ax.axis('off')
                
                # Visualize Sequence space
                elif isinstance(space, gymnasium.spaces.Sequence):
                    ax.text(0.5, 0.5, f"{title}\nSequence Space\nBase Space: {space.feature_space}\nObservation Length: {len(obs)}", 
                            ha='center', va='center')
                    ax.axis('off')
                
                # Visualize Graph space
                elif isinstance(space, gymnasium.spaces.Graph):
                    if isinstance(obs, gymnasium.spaces.GraphInstance):
                        ax.text(0.5, 0.5, f"{title}\nGraph Space\nNodes: {obs.nodes.shape}\nEdges: {obs.edges.shape if obs.edges is not None else 'None'}\nEdge Links: {obs.edge_links.shape}", 
                                ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, f"{title}\nGraph Space\nObservation: {obs}", 
                                ha='center', va='center')
                    ax.axis('off')
                
                # Visualize Unsupported space type
                else:
                    ax.text(0.5, 0.5, f"{title}\nUnsupported Space Type: {type(space)}\nObservation: {obs}", 
                            ha='center', va='center')
                    ax.axis('off')
            
            # Handle different types of observation spaces
            if isinstance(env.observation_space, gymnasium.spaces.Dict):
                rows = len(env.observation_space.spaces)
                fig, axs = plt.subplots(rows, 1, figsize=(8, 6*rows))
                for i, (key, space) in enumerate(env.observation_space.spaces.items()):
                    visualize_space(space, observation[key], axs[i] if rows > 1 else axs, title=f"{key}")
            elif isinstance(env.observation_space, gymnasium.spaces.Tuple):
                rows = len(env.observation_space.spaces)
                fig, axs = plt.subplots(rows, 1, figsize=(8, 6*rows))
                for i, (space, obs) in enumerate(zip(env.observation_space.spaces, observation)):
                    visualize_space(space, obs, axs[i] if rows > 1 else axs, title=f"Element {i}")
            else:
                visualize_space(env.observation_space, observation, ax)
            
            # Create and display the matplotlib canvas
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Unable to visualize environment: {str(e)}")
    else:
        messagebox.showwarning("No Environment Selected", "Please select an environment from the list.")

# Create the main window
window = tk.Tk()
window.title("Gymnasium Environment Explorer")

# Create and pack the main frame
frame = ttk.Frame(window)
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)

# Create a label for the search entry
search_label = ttk.Label(frame, text="Search Environments:")
search_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

# Create a StringVar for the search entry and trace it to update the listbox
search_var = tk.StringVar()
search_var.trace("w", search_listbox)
search_entry = ttk.Entry(frame, textvariable=search_var, width=50)
search_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

# Create a label for the listbox
listbox_label = ttk.Label(frame, text="Available Environments:")
listbox_label.grid(row=2, column=0, sticky=tk.W)

# Create the listbox frame
listbox_frame = ttk.Frame(frame)
listbox_frame.grid(row=3, column=0, sticky=(tk.N, tk.S, tk.W))
frame.rowconfigure(3, weight=1)

# Set the listbox maximum width to the maximum length of the environment names
max_width = max(len(env_id) for env_id in gymnasium.envs.registration.registry) + 2  # Add 2 for scrollbar padding

# Create and pack the listbox
listbox = tk.Listbox(listbox_frame, width=max_width)
listbox.pack(side=tk.LEFT, fill=tk.Y)

# Create and pack a scrollbar for the listbox
listbox_scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
listbox.config(yscrollcommand=listbox_scrollbar.set)

# Create a label for the environment details
details_label = ttk.Label(frame, text="Environment Details:")
details_label.grid(row=2, column=1, sticky=tk.W)

# Text widget and its scrollbar
text_frame = ttk.Frame(frame)
text_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(1, weight=1)

# Create and pack a text widget to display environment specifications
env_spec_text = tk.Text(text_frame, wrap=tk.WORD, width=1) # Set a minimum width the scrollbar will appear at
env_spec_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create and pack a scrollbar for the text widget
text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=env_spec_text.yview)
text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
env_spec_text.config(yscrollcommand=text_scrollbar.set)

# Create a frame for the buttons
button_frame = ttk.Frame(frame)
button_frame.grid(row=4, column=0, columnspan=2, pady=10)

# Add Play button
play_button = ttk.Button(button_frame, text="Play Selected Environment", command=play_environment)
play_button.grid(row=0, column=0, padx=(0, 5))

# Add Random Agent button
random_agent_button = ttk.Button(button_frame, text="Run Random Agent", command=run_random_agent_for_environment)
random_agent_button.grid(row=0, column=1, padx=(5, 5))

# Add Visualize button
visualize_button = ttk.Button(button_frame, text="Visualize Observation Space", command=visualize_environment)
visualize_button.grid(row=0, column=2, padx=(5, 0))

# Configure the columns in the button frame to have equal weight
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)
button_frame.columnconfigure(2, weight=1)

# Populate the listbox initially
search_listbox()

# Bind the selection event of the listbox to the get_env_spec function
listbox.bind('<<ListboxSelect>>', get_env_spec)

# Set initial window size
window.geometry("800x600")

# Start the Tkinter event loop
window.mainloop()
