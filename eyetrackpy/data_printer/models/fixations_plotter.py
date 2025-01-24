import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors


class FixationsPlotter:
    """
    expected input:
    words_with_numbers = [
    ("in", 0),
    ("1958", 58),
    ("clooney", 100),
    ]"""

    @staticmethod
    def plot_fixations(words_with_numbers, save_path=None, add_numbers=False):
        # Define the number of columns for the grid
        n_columns = 10

        # Determine the number of rows based on the number of columns
        n_rows = (len(words_with_numbers) + n_columns - 1) // n_columns

        # Create arrays for words and numbers
        words = np.array([word for word, _ in words_with_numbers])
        numbers = np.array([number for _, number in words_with_numbers])

        # Pad the arrays to fit the grid shape if necessary
        words = np.pad(
            words,
            (0, n_rows * n_columns - len(words)),
            mode="constant",
            constant_values="",
        )
        numbers = np.pad(
            numbers,
            (0, n_rows * n_columns - len(numbers)),
            mode="constant",
            constant_values=0,
        )

        # Reshape the arrays into a grid
        words_grid = words.reshape((n_rows, n_columns))
        numbers_grid = numbers.reshape((n_rows, n_columns))

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(15, n_rows))

        # Define a color map for the numbers
        cmap = plt.cm.get_cmap("Oranges")

        # Normalize the numbers for color mapping
        norm = plt.Normalize(numbers.min(), numbers.max())
        rectangle_height = 0.75
        # Add text and colored background
        for i in range(n_rows):
            for j in range(n_columns):
                word = words_grid[i, j]
                number = numbers_grid[i, j]
                color = cmap(norm(number))
                # Add colored rectangle
                ax.add_patch(
                    plt.Rectangle(
                        (j, n_rows - i - 1),
                        1,
                        rectangle_height,
                        color=color,
                        ec="black",
                    )
                )
                # Add text
                ax.text(
                    j + 0.5,
                    n_rows - i - 1 + rectangle_height / 2 + 0.1,
                    word,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                if add_numbers:
                    ax.text(
                        j + 0.5,
                        n_rows - i - 1 + rectangle_height / 2 - 0.2,
                        "(" + str(round(number, 5)) + ")",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        # Remove axes
        ax.set_xlim(0, n_columns)
        ax.set_ylim(0, n_rows)
        # ax.set_aspect('equal')
        ax.set_aspect("auto")
        ax.axis("off")

        # Show the plot
        # Save the plot as a high-resolution PNG file
        if save_path is not None:
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight"
            )  # Save with better layout

        plt.show()

    @staticmethod
    def split_words_across_lines(words, values, max_char_per_line=60):
        # Splitting the words and values into lines
        lines = []
        line_values = []
        words_line_character = 0
        words_line_id = []
        # print(words)
        for i in range(0, len(words)):
            # print(words_line_character + len(words[i]))
            if words_line_character + len(words[i]) > max_char_per_line:
                words_line_character += len(words[i]) + 1
                words_line_id.append(i)
                lines.append([words[j] for j in words_line_id])
                line_values.append([values[j] for j in words_line_id])
                words_line_character = 0
                words_line_id = []
            else:
                words_line_character += len(words[i]) + 1
                words_line_id.append(i)
        lines.append([words[j] for j in words_line_id])
        line_values.append([values[j] for j in words_line_id])
        # lines = [words[i:i + max_words_per_line] for i in range(0, len(words), max_words_per_line)]
        # line_values = [values[i:i + max_words_per_line] for i in range(0, len(values), max_words_per_line)]
        return lines, line_values

    @staticmethod
    def plot_fixations(words_with_numbers, max_char_per_line=60, save_path=None):
        # Set up the plot
        words = [x[0] for x in words_with_numbers]
        values = [x[1] for x in words_with_numbers]
        # Set up the plot dimensions
        # Function to split words across lines
        # def split_words_across_lines(words, values, max_words_per_line):
        #     # Splitting the words and values into lines
        #     lines = [words[i:i + max_words_per_line] for i in range(0, len(words), max_words_per_line)]
        #     line_values = [values[i:i + max_words_per_line] for i in range(0, len(values), max_words_per_line)]
        #     return lines, line_values

        # Get the words and values split into lines
        lines, line_values = FixationsPlotter.split_words_across_lines(
            words, values, max_char_per_line=max_char_per_line
        )
        # print(lines)
        # max_num_words = max([len(line) for line in lines])
        # Set up the plot with enough height for multiple lines
        fig, ax = plt.subplots(figsize=(8, len(lines) / 2))

        # Create a colormap (blue for low values, red for high values)
        cmap = sns.color_palette("light:#5A9_r", as_cmap=True)

        # Normalize values between 0 and 1
        norm = colors.Normalize(vmin=min(values), vmax=max(values))

        # Plot each word with its corresponding color on multiple lines
        max_i = []
        for line_num, (line, vals) in enumerate(zip(lines, line_values)):
            prev_i = 0
            prev_adding_space = 0.05
            for i, (word, value) in enumerate(zip(line, vals)):
                num_chars = len(word)  # Get the number of characters in the word
                # Example base value (can be decimal)
                scale_factor = 0.1  # Example scale factor (can also be decimal)
                # Adjust 'i' based on the number of characters (allowing for decimal values)
                adding_space = 0.1 + (num_chars) * scale_factor
                i = prev_i + prev_adding_space + adding_space
                # print(prev_i, i)
                ax.text(
                    i,
                    -line_num,
                    word,
                    fontsize=12,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor=cmap(norm(value)),
                        edgecolor="none",
                        boxstyle="round,pad=0.3",
                    ),
                )
                prev_i = i
                prev_adding_space = adding_space
                max_i.append(i + adding_space)
        # Set limits and hide axes
        # ax.set_xlim(-1, i)
        ax.set_xlim(-1, max(max_i))
        ax.set_ylim(-len(lines), 1)
        ax.axis("off")

        # Show the plot
        if save_path is not None:
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight"
            )  # Save with better layout

        plt.tight_layout()
        plt.show()
