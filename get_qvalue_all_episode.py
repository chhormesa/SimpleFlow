import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_q_values(model_num):
    # Paths
    csv_path = os.path.join('results', str(model_num), 'q_values.csv')
    pdf_path = csv_path.replace('.csv', '.pdf')

    # Load full data
    df = pd.read_csv(csv_path)

    # Create numeric x-axis for plotting
    df['x'] = range(len(df))

    # Plot all data
    plt.figure(figsize=(16, 9))
    plt.plot(df['x'], df['Q_keep'], label='Action 0: Keep', linewidth=0.5)
    plt.plot(df['x'], df['Q_switch'], label='Action 1: Switch', linewidth=0.5)

    # Labels and style
    plt.xlabel('Total Steps')
    plt.ylabel('Q-value')
    plt.title('Q-values for All Episodes and Steps')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    # Save
    plt.savefig(pdf_path, format='pdf')
    print(f"Plot saved to {pdf_path}")
    plt.close()

if __name__ == '__main__':
    plot_q_values(321)
