def save_plot(fig, file_name):
    """Saves a plot to the specified file."""
    fig.savefig(file_name)
    print(f"Plot saved as {file_name}")
