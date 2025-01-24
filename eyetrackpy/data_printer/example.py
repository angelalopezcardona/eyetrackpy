from models.fixations_plotter import FixationsPlotter

words_with_numbers = [
    ("in", 0),
    ("1958", 58),
    ("clooney", 100),
    ("left", 20),
    ("columbia", 80),
    ("doing", 10),
    ("a", 5),
    ("number", 50),
    ("of", 15),
    ("recordings", 75),
    ("for", 10),
    ("mgm", 60),
    ("records", 55),
    ("and", 25),
    ("then", 30),
    ("some", 35),
    ("for", 10),
    ("coral", 40),
    ("records", 55),
]
# plot_fixations(words_with_numbers)
FixationsPlotter.plot_fixations(words_with_numbers)
