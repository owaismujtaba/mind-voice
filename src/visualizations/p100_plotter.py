import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

class P100Plotter:
    def __init__(self, 
            condition1: 'P100ComponentAnalyzer', condition2: 'P100ComponentAnalyzer', 
            name1, name2, sub_id, ses_id, config, logger
            ):
        """
        Initialize the P100 plotter.

        Args:
            condition1 (P100ComponentAnalyzer): First condition analyzer.
            condition2 (P100ComponentAnalyzer): Second condition analyzer.
        """
        self.condition1 = condition1
        self.condition2 = condition2
        self.name1 = name1
        self.name2 = name2
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.logger = logger
        self.config = config
        self.logger.info('Initializing Plotter')


    def plot_evokeds(self):
        """
        Plot the average evoked responses across all selected channels for condition1 and condition2.
        Converts data to microvolts, includes channel names in legend, highlights 80–120 ms window,
        and adds a vertical line at 0 ms.
        """
        self.logger.info('Plotting Evoked')
        evoked_1 = self.condition1.get_evoked()
        evoked_2 = self.condition2.get_evoked()

        ch_names_1 = self.condition1.channels
        ch_names_2 = self.condition2.channels

        try:
            ch_idx_1 = [evoked_1.ch_names.index(ch) for ch in ch_names_1]
            ch_idx_2 = [evoked_2.ch_names.index(ch) for ch in ch_names_2]
        except ValueError as e:
            raise ValueError(f"Channel not found: {e}")

        data_1 = evoked_1.data[ch_idx_1, :] * 1e6  # Convert to microvolts
        data_2 = evoked_2.data[ch_idx_2, :] * 1e6  # Convert to microvolts

        min_len = min(data_1.shape[1], data_2.shape[1])
        times = evoked_1.times[:min_len]

        data_1 = data_1[:, :min_len]
        data_2 = data_2[:, :min_len]

        # Compute average across channels
        mean_1 = data_1.mean(axis=0)
        mean_2 = data_2.mean(axis=0)

        # Channel names in legend
        ch_str_1 = ', '.join(ch_names_1)
        ch_str_2 = ', '.join(ch_names_2)

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))

        # Highlight 80–120 ms window
        plt.axvspan(0.080, 0.120, color='orange', alpha=0.3, label='80–120 ms window')

        # Add vertical line at 0 ms
        plt.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Stimulus Onset')

        # Plot evoked responses
        plt.plot(times, mean_1, label=f'{self.name1} (mean) [{ch_str_1}]', linestyle='-', linewidth=2.0)
        plt.plot(times, mean_2, label=f'{self.name2} (mean) [{ch_str_2}]', linestyle='--', linewidth=2.0)

        # Axis labels and aesthetics
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude (µV)', fontsize=12)
        plt.legend(loc='best', fontsize=9, frameon=True)
        plt.grid(True, which='major', linestyle='--', alpha=0.6)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save figure
        directory = Path(os.getcwd(), self.config['analysis']['results_dir'], 'P100', 'Plots')
        #directory = Path(directory, f'sub-{self.sub_id}', f'ses-{self.ses_id}')
        os.makedirs(directory, exist_ok=True)
        plot_name =f'sub-{self.sub_id}_ses-{self.ses_id}_p-100-{self.name1}_{self.name2}_mean.png'
        filepath = Path(directory, plot_name)

        plt.savefig(filepath, dpi=300)
        plt.close()


    