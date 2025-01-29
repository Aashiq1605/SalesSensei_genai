import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for rendering plots in memory
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

def generate_graph(graph_type, graph_data, title, x_label, y_label):
    try:
        # Convert graph data to a DataFrame
        df = pd.DataFrame(graph_data)

        # Check if x_label and y_label are present in the DataFrame
        if x_label not in df.columns or y_label not in df.columns:
            raise ValueError(f"Columns '{x_label}' or '{y_label}' not found in data.")

        # Data cleaning: Group by X-axis values and aggregate Y-axis values
        df = df.groupby(x_label, as_index=False)[y_label].sum()

        # Log the cleaned data
        print(f"Cleaned Data:\n{df}")

        # Create the plot
        plt.figure(figsize=(12, 6))  # Increased size for better visibility
        ax = None

        if graph_type == "bar":
            ax = df.plot.bar(x=x_label, y=y_label, legend=False)

            # Add data labels on bars
            for i, v in enumerate(df[y_label]):
                ax.text(i, v + 1, str(round(v, 2)), ha='center', fontsize=10, fontweight='bold')

        elif graph_type == "line":
            ax = df.plot.line(x=x_label, y=y_label, legend=False, marker='o')

            # Add data labels on points
            for i, v in enumerate(df[y_label]):
                ax.text(i, v + 1, str(round(v, 2)), ha='center', fontsize=10, fontweight='bold')

        elif graph_type == "scatter":
            plt.scatter(df[x_label], df[y_label])

            # Add data labels on scatter points
            for i, v in enumerate(df[y_label]):
                plt.text(df[x_label][i], v + 1, str(round(v, 2)), ha='center', fontsize=10, fontweight='bold')

        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

        # Rotate X-axis labels for better visibility
        plt.xticks(rotation=45, ha="right")

        # Add title and labels
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Save the plot to a buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        # Return the graph as a Base64 string
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Error in generate_graph: {e}")
        raise

