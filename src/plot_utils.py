import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cf_matrix, labels, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(cf_matrix, cmap=cmap, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    return fig