
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def generate_plot(data, plot_type='bar', x=None, y=None, title='', xlabel='', ylabel='', hue=None, filename=None):
    """
    Generate different types of plots using seaborn.
    
    Parameters:
    - data: DataFrame or array-like, the dataset to plot.
    - plot_type: str, the type of plot ('bar', 'hist', 'scatter', 'line', etc.).
    - x: str, the column name for the x-axis (if applicable).
    - y: str, the column name for the y-axis (if applicable).
    - title: str, the title of the plot.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - hue: str, column to group data by color (used in scatter plots and barplots).
    
    Supported plot types: 'bar', 'hist', 'scatter', 'line', 'count'.
    """
    plt.figure(figsize=(len(x)*2, len(y)*1))
    
    # Bar Plot
    if plot_type == 'bar':
        sns.barplot(x=x, y=y, hue=hue, data=data)
        
    # Histogram
    elif plot_type == 'hist':
        sns.histplot(data[x], kde=True)
        
    # Scatter Plot
    elif plot_type == 'scatter':
        sns.scatterplot(x=x, y=y, hue=hue, data=data)
        
    # Line Plot
    elif plot_type == 'line':
        sns.lineplot(x=x, y=y, hue=hue, data=data)
        
    # Count Plot (for categorical columns)
    elif plot_type == 'count':
        sns.countplot(x=x, data=data, hue=hue)
    
    # Set plot details
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # dpi controls resolution, bbox_inches prevents clipping

    # Show the plot
    plt.show()
