import gymnasium
import tkinter as tk
from tkinter import ttk
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

# Populate the listbox initially
search_listbox()

# Bind the selection event of the listbox to the get_env_spec function
listbox.bind('<<ListboxSelect>>', get_env_spec)

# Set initial window size
window.geometry("800x600")

# Start the Tkinter event loop
window.mainloop()