"""
Task Accuracy Tracker for Continual Learning

This module provides functionality to track task accuracies over time during
continual learning and generate plots showing the accuracy trends of all tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict

class TaskAccuracyTracker:
    """
    Tracks and visualizes task accuracies over time during continual learning.
    """
    
    def __init__(self, num_tasks: int, save_dir: str = "./plots"):
        """
        Initialize the task accuracy tracker.
        
        Args:
            num_tasks: Total number of tasks in the continual learning sequence
            save_dir: Directory to save the accuracy plots
        """
        self.num_tasks = num_tasks
        self.save_dir = save_dir
        self.task_accuracies = {}  # Dict[task_id, List[accuracy_values]]
        self.task_names = {}  # Dict[task_id, task_name]
        self.current_task = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize accuracy storage for each task
        for task_id in range(num_tasks):
            self.task_accuracies[task_id] = []
            self.task_names[task_id] = f"Task {task_id + 1}"
    
    def update_accuracies(self, task_results: List[float], current_task_id: int):
        """
        Update accuracies for all tasks evaluated so far.
        
        Args:
            task_results: List of accuracies for tasks [0, 1, ..., current_task_id]
            current_task_id: The current task being trained (0-indexed)
        """
        self.current_task = current_task_id
        
        # Update accuracies for all tasks evaluated
        for task_id, accuracy in enumerate(task_results):
            if task_id <= current_task_id:
                self.task_accuracies[task_id].append(accuracy)
        
        # For tasks not yet introduced, add None placeholders
        for task_id in range(len(task_results), self.num_tasks):
            if len(self.task_accuracies[task_id]) < len(self.task_accuracies[0]):
                self.task_accuracies[task_id].append(None)
    
    def set_task_name(self, task_id: int, name: str):
        """
        Set a custom name for a task.
        
        Args:
            task_id: Task identifier (0-indexed)
            name: Custom name for the task
        """
        if task_id < self.num_tasks:
            self.task_names[task_id] = name
    
    def plot_accuracy_trends(self, 
                           title: str = "Task Accuracies Over Time",
                           filename: str = "task_accuracy_trends.png",
                           figsize: tuple = (12, 8),
                           save_plot: bool = True,
                           show_plot: bool = False) -> None:
        """
        Create and save a plot showing accuracy trends for all tasks.
        
        Args:
            title: Title for the plot
            filename: Filename to save the plot
            figsize: Figure size (width, height)
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
        """
        plt.figure(figsize=figsize)
        
        # Set up color palette
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_tasks))
        
        # Plot accuracy trends for each task
        for task_id in range(self.num_tasks):
            accuracies = self.task_accuracies[task_id]
            
            # Filter out None values and create corresponding x-axis values
            valid_accuracies = [acc for acc in accuracies if acc is not None]
            if not valid_accuracies:
                continue
                
            # X-axis represents evaluation points (after each task completion)
            x_values = list(range(task_id + 1, task_id + 1 + len(valid_accuracies)))
            
            plt.plot(x_values, valid_accuracies, 
                    marker='o', linewidth=2, markersize=6,
                    color=colors[task_id], 
                    label=self.task_names[task_id])
        
        # Customize plot
        plt.xlabel('Task Completion (Training Sequence)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to show task completion points
        max_tasks_evaluated = max(len([acc for acc in accs if acc is not None]) 
                                for accs in self.task_accuracies.values())
        plt.xticks(range(1, max_tasks_evaluated + 1), 
                  [f"After Task {i}" for i in range(1, max_tasks_evaluated + 1)])
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent clipping of labels
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            plot_path = os.path.join(self.save_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Task accuracy trends plot saved to: {plot_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_forgetting_analysis(self, 
                                filename: str = "forgetting_analysis.png",
                                figsize: tuple = (12, 8),
                                save_plot: bool = True,
                                show_plot: bool = False) -> None:
        """
        Create a plot showing forgetting analysis for each task.
        
        Args:
            filename: Filename to save the plot
            figsize: Figure size (width, height)
            save_plot: Whether to save the plot to file
            show_plot: Whether to display the plot
        """
        plt.figure(figsize=figsize)
        
        forgetting_data = []
        task_labels = []
        
        for task_id in range(self.num_tasks):
            accuracies = [acc for acc in self.task_accuracies[task_id] if acc is not None]
            if len(accuracies) > 1:
                # Calculate forgetting as max_accuracy - final_accuracy
                max_acc = max(accuracies)
                final_acc = accuracies[-1]
                forgetting = max_acc - final_acc
                forgetting_data.append(forgetting)
                task_labels.append(self.task_names[task_id])
        
        if forgetting_data:
            colors = plt.cm.Set1(np.linspace(0, 1, len(forgetting_data)))
            bars = plt.bar(task_labels, forgetting_data, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, value in zip(bars, forgetting_data):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel('Tasks', fontsize=12)
            plt.ylabel('Forgetting (Max Acc - Final Acc) %', fontsize=12)
            plt.title('Catastrophic Forgetting Analysis', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
        
        # Save plot
        if save_plot:
            plot_path = os.path.join(self.save_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Forgetting analysis plot saved to: {plot_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of task performance.
        
        Returns:
            Dictionary containing summary statistics
        """
        stats = {
            'final_accuracies': {},
            'max_accuracies': {},
            'forgetting': {},
            'mean_final_accuracy': 0.0,
            'mean_forgetting': 0.0
        }
        
        final_accs = []
        forgetting_values = []
        
        for task_id in range(self.num_tasks):
            accuracies = [acc for acc in self.task_accuracies[task_id] if acc is not None]
            if accuracies:
                final_acc = accuracies[-1]
                max_acc = max(accuracies)
                forgetting = max_acc - final_acc
                
                stats['final_accuracies'][self.task_names[task_id]] = final_acc
                stats['max_accuracies'][self.task_names[task_id]] = max_acc
                stats['forgetting'][self.task_names[task_id]] = forgetting
                
                final_accs.append(final_acc)
                forgetting_values.append(forgetting)
        
        if final_accs:
            stats['mean_final_accuracy'] = np.mean(final_accs)
        if forgetting_values:
            stats['mean_forgetting'] = np.mean(forgetting_values)
        
        return stats
    
    def save_data(self, filename: str = "task_accuracy_data.npz"):
        """
        Save the tracked data to a file.
        
        Args:
            filename: Filename to save the data
        """
        save_path = os.path.join(self.save_dir, filename)
        np.savez(save_path, 
                task_accuracies=self.task_accuracies,
                task_names=self.task_names,
                num_tasks=self.num_tasks)
        print(f"Task accuracy data saved to: {save_path}")
    
    def load_data(self, filename: str = "task_accuracy_data.npz"):
        """
        Load previously saved tracking data.
        
        Args:
            filename: Filename to load the data from
        """
        load_path = os.path.join(self.save_dir, filename)
        if os.path.exists(load_path):
            data = np.load(load_path, allow_pickle=True)
            self.task_accuracies = data['task_accuracies'].item()
            self.task_names = data['task_names'].item()
            self.num_tasks = data['num_tasks'].item()
            print(f"Task accuracy data loaded from: {load_path}")
        else:
            print(f"No saved data found at: {load_path}")