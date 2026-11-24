import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, Wedge

# Cell (row, col) is drawn in the square [col, col + 1] x [row, row + 1].
# The y axis is inverted so that row 0 is at the top, like in the grid array.

GRASS_COLOR = np.array([0.42, 0.66, 0.33])
GRASS_DARK = (0.28, 0.50, 0.22)
OUTLINE_COLOR = (0.22, 0.20, 0.20)
WOOD_COLOR = (0.62, 0.43, 0.22)
WOOD_LIGHT = (0.76, 0.58, 0.34)
WOOD_DARK = (0.34, 0.22, 0.10)
MUD_COLOR = (0.55, 0.38, 0.22)
MUD_DARK = (0.38, 0.25, 0.13)
STEEL_COLOR = (0.38, 0.38, 0.42)
STEEL_LIGHT = (0.86, 0.86, 0.89)
RUST_COLOR = (0.62, 0.30, 0.12)
STONE_COLOR = (0.84, 0.81, 0.72)
STONE_EDGE = (0.55, 0.50, 0.40)


def cell_random(row, col):
    return np.random.RandomState(row * 100 + col)


def draw_grass(ax, row, col):
    rng = cell_random(row, col)
    shade = rng.uniform(-0.04, 0.04)
    color = np.clip(GRASS_COLOR + shade, 0, 1)
    ax.add_patch(Rectangle((col, row), 1, 1, facecolor=color, edgecolor="none", zorder=1))

    # A few small blades of grass.
    for _ in range(4):
        x = col + rng.uniform(0.15, 0.85)
        y = row + rng.uniform(0.35, 0.90)
        ax.plot([x, x + 0.04], [y, y - 0.14], color=GRASS_DARK, linewidth=0.8, zorder=2)


def draw_fence(ax, row, col):
    # The rails run over the whole width of the cell, so that neighbouring
    # fence cells join into one continuous barrier.
    shadow = Rectangle((col, row + 0.80), 1, 0.08, facecolor=(0, 0, 0, 0.25), edgecolor="none", zorder=3)
    ax.add_patch(shadow)

    for rail_top in [row + 0.30, row + 0.58]:
        rail = Rectangle((col, rail_top), 1, 0.12, facecolor=WOOD_COLOR, edgecolor=WOOD_DARK, linewidth=1.0, zorder=4)
        ax.add_patch(rail)
        highlight = Rectangle((col, rail_top + 0.015), 1, 0.03, facecolor=WOOD_LIGHT, edgecolor="none", zorder=5)
        ax.add_patch(highlight)

    for post_x in [col + 0.25, col + 0.75]:
        post = Rectangle((post_x - 0.07, row + 0.18), 0.14, 0.66, facecolor=WOOD_COLOR, edgecolor=WOOD_DARK, linewidth=1.0, zorder=6)
        ax.add_patch(post)
        ax.add_patch(Rectangle((post_x - 0.07, row + 0.18), 0.14, 0.05, facecolor=WOOD_LIGHT, edgecolor="none", zorder=7))


def draw_mud(ax, row, col):
    rng = cell_random(row, col)
    ax.add_patch(Rectangle((col, row), 1, 1, facecolor=MUD_COLOR, edgecolor="none", zorder=1))

    for _ in range(4):
        x = col + rng.uniform(0.2, 0.8)
        y = row + rng.uniform(0.2, 0.8)
        puddle_width = rng.uniform(0.18, 0.34)
        puddle_height = puddle_width * 0.55
        ax.add_patch(Ellipse((x, y), puddle_width, puddle_height, facecolor=MUD_DARK, edgecolor="none", zorder=2))


def draw_jaw_teeth(ax, center_x, center_y, first_angle, last_angle):
    """Teeth along the inner edge of one jaw, pointing towards the center."""
    inner_radius = 0.30
    tip_radius = 0.14
    half_width = np.radians(8)

    angle = first_angle
    while angle <= last_angle:
        radians = np.radians(angle)
        base_one = (center_x + inner_radius * np.cos(radians - half_width), center_y + inner_radius * np.sin(radians - half_width))
        base_two = (center_x + inner_radius * np.cos(radians + half_width), center_y + inner_radius * np.sin(radians + half_width))
        tip = (center_x + tip_radius * np.cos(radians), center_y + tip_radius * np.sin(radians))
        tooth = Polygon([base_one, base_two, tip], closed=True, facecolor=STEEL_LIGHT, edgecolor=OUTLINE_COLOR, linewidth=0.6, zorder=6)
        ax.add_patch(tooth)
        angle += 18


def draw_trap(ax, row, col):
    """An open bear trap seen from above: two toothed jaws and a pressure plate."""
    center_x = col + 0.5
    center_y = row + 0.5

    ax.add_patch(Circle((center_x + 0.03, center_y + 0.05), 0.42, facecolor=(0, 0, 0, 0.25), edgecolor="none", zorder=3))

    # Dark ground inside the open jaws, so that the teeth stand out.
    ax.add_patch(Circle((center_x, center_y), 0.31, facecolor=(0.14, 0.11, 0.09), edgecolor="none", zorder=4))

    # Left and right jaws, with a wide gap between them at the top and bottom.
    left_jaw = Wedge((center_x - 0.03, center_y), 0.43, 110, 250, width=0.13, facecolor=STEEL_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.9, zorder=5)
    right_jaw = Wedge((center_x + 0.03, center_y), 0.43, -70, 70, width=0.13, facecolor=STEEL_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.9, zorder=5)
    ax.add_patch(left_jaw)
    ax.add_patch(right_jaw)

    draw_jaw_teeth(ax, center_x - 0.03, center_y, 122, 238)
    draw_jaw_teeth(ax, center_x + 0.03, center_y, -58, 58)

    # Small plate in the middle that triggers the trap.
    plate = Rectangle((center_x - 0.09, center_y - 0.06), 0.18, 0.12, facecolor=RUST_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.7, zorder=7)
    ax.add_patch(plate)


def draw_start(ax, row, col):
    center_x = col + 0.5
    center_y = row + 0.5
    ax.add_patch(Circle((center_x, center_y), 0.40, facecolor=STONE_COLOR, edgecolor=STONE_EDGE, linewidth=1.5, zorder=3))
    ax.add_patch(Circle((center_x, center_y), 0.26, facecolor="none", edgecolor=STONE_EDGE, linewidth=1.0, zorder=4))


def draw_goal(ax, row, col):
    pole_x = col + 0.36
    pole_bottom = row + 0.88
    pole_top = row + 0.10

    ax.add_patch(Ellipse((pole_x, pole_bottom), 0.30, 0.10, facecolor=(0, 0, 0, 0.25), edgecolor="none", zorder=3))
    ax.plot([pole_x, pole_x], [pole_bottom, pole_top], color=WOOD_DARK, linewidth=2.2, zorder=4)

    flag = Polygon(
        [(pole_x, pole_top), (pole_x + 0.50, pole_top + 0.17), (pole_x, pole_top + 0.34)],
        closed=True,
        facecolor=(0.90, 0.18, 0.18),
        edgecolor=(0.50, 0.08, 0.08),
        linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(flag)
    ax.add_patch(Circle((pole_x, pole_top), 0.04, facecolor=(0.95, 0.80, 0.20), edgecolor="none", zorder=6))


def draw_map(ax, grid, start, goal):
    grid = np.array(grid)
    size = grid.shape[0]

    for row in range(size):
        for col in range(size):
            value = grid[row, col]
            if value == 3:
                draw_mud(ax, row, col)
            else:
                draw_grass(ax, row, col)

            if value == 1:
                draw_fence(ax, row, col)
            elif value == 2:
                draw_trap(ax, row, col)

    draw_start(ax, start[0], start[1])
    draw_goal(ax, goal[0], goal[1])

    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(OUTLINE_COLOR)
        side.set_linewidth(2)


def draw_legend(ax):
    """One tile per kind of cell, with its name below."""
    names = ["Obstacle", "Trap", "Mud", "Start", "Goal"]

    # One empty cell between two tiles, so that the names do not touch.
    for index, name in enumerate(names):
        col = 2 * index

        if name == "Mud":
            draw_mud(ax, 0, col)
        else:
            draw_grass(ax, 0, col)

        if name == "Obstacle":
            draw_fence(ax, 0, col)
        elif name == "Trap":
            draw_trap(ax, 0, col)
        elif name == "Start":
            draw_start(ax, 0, col)
        elif name == "Goal":
            draw_goal(ax, 0, col)

        ax.text(col + 0.5, 1.30, name, ha="center", va="center", fontsize=11)

    ax.set_xlim(-0.2, 2 * len(names) - 1 + 0.2)
    ax.set_ylim(1.6, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_agent(ax, position, scale=1.35):
    """Draw a small robot on a cell and return its artists, so that the
    caller can remove them before drawing the next position."""
    row, col = position
    center_x = col + 0.5
    center_y = row + 0.5
    s = scale
    artists = []

    shadow = Ellipse((center_x, center_y + 0.32 * s), 0.52 * s, 0.14 * s, facecolor=(0, 0, 0, 0.30), edgecolor="none", zorder=8)
    body = Rectangle((center_x - 0.25 * s, center_y - 0.20 * s), 0.50 * s, 0.46 * s, facecolor=(0.20, 0.45, 0.85), edgecolor=(0.08, 0.20, 0.50), linewidth=1.3, zorder=9)
    left_eye = Circle((center_x - 0.10 * s, center_y - 0.04 * s), 0.075 * s, facecolor="white", edgecolor="none", zorder=10)
    right_eye = Circle((center_x + 0.10 * s, center_y - 0.04 * s), 0.075 * s, facecolor="white", edgecolor="none", zorder=10)
    left_pupil = Circle((center_x - 0.09 * s, center_y - 0.03 * s), 0.035 * s, facecolor="black", edgecolor="none", zorder=11)
    right_pupil = Circle((center_x + 0.11 * s, center_y - 0.03 * s), 0.035 * s, facecolor="black", edgecolor="none", zorder=11)

    for patch in [shadow, body, left_eye, right_eye, left_pupil, right_pupil]:
        ax.add_patch(patch)
        artists.append(patch)

    antenna, = ax.plot([center_x, center_x], [center_y - 0.20 * s, center_y - 0.34 * s], color=(0.08, 0.20, 0.50), linewidth=1.5, zorder=9)
    tip = Circle((center_x, center_y - 0.36 * s), 0.045 * s, facecolor=(0.95, 0.80, 0.20), edgecolor="none", zorder=10)
    ax.add_patch(tip)
    artists.append(antenna)
    artists.append(tip)

    return artists


def remove_artists(artists):
    for artist in artists:
        artist.remove()


def trail_points(positions, last_index):
    """Coordinates of the path up to last_index. A trap sends the agent back
    to the start, so the line is cut there instead of crossing the map."""
    xs = []
    ys = []
    for index in range(last_index + 1):
        row, col = positions[index]
        if index > 0:
            previous_row, previous_col = positions[index - 1]
            distance = abs(row - previous_row) + abs(col - previous_col)
            if distance > 1:
                xs.append(np.nan)
                ys.append(np.nan)
        xs.append(col + 0.5)
        ys.append(row + 0.5)
    return xs, ys
