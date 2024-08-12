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
    selected_env = listbox.get(listbox.curselection())  # Get the selected environment
    env_spec = gymnasium.spec(selected_env)  # Get the environment specifications
    env_spec_text.delete(1.0, tk.END)  # Clear the current text
    # Display each specification in the text widget
    [env_spec_text.insert(tk.END, f"{key}: {value}\n") for key, value in env_spec.__dict__.items()]

# Create the main window
window = tk.Tk()

# Create and pack the main frame
frame = ttk.Frame(window)
frame.pack()

# Create a StringVar for the search entry and trace it to update the listbox
search_var = tk.StringVar()
search_var.trace("w", search_listbox)
search_entry = ttk.Entry(frame, textvariable=search_var, width=50)
search_entry.pack()

# Create and pack the listbox
listbox = tk.Listbox(frame, height=20, width=50)
listbox.pack(side=tk.LEFT, fill=tk.BOTH)

# Create and pack a scrollbar for the listbox
scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Connect the scrollbar to the listbox
listbox.config(yscrollcommand=scrollbar.set)

# Populate the listbox initially
search_listbox()

# Create and pack a text widget to display environment specifications
env_spec_text = tk.Text(frame, height=20, width=50)
env_spec_text.pack(side=tk.RIGHT)

# Bind the selection event of the listbox to the get_env_spec function
listbox.bind('<<ListboxSelect>>', get_env_spec)

# Start the Tkinter event loop
window.mainloop()