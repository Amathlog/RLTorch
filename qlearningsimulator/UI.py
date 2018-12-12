from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QLabel, QDialog, QWidget
from PyQt5.QtGui import QPainter, QColor

import numpy as np

from qlearningsimulator.qlearning import Action, Grid, QLearning

TEXT = "In this simple 4x3 grid-world, Q-learning agent learns by trial and error from interactions with the \n" \
       "environment. Agent starts the episode in the bottom left corner of the maze. The goal of the agent is to \n" \
       "maximize its total (future) reward. It does this by learning which action is optimal for each state. The \n" \
       "action that is optimal for each state is the action that has the highest long-term reward. Episode \n" \
       "terminates when the agent reaches +1 or -1 state, in all other states agent, will receive an immediate \n" \
       "reward -0.1. If the agent enters the wall, it bounces back. At the beginning of each game, all Q values \n" \
       "are set to zero."

WINDOW_TITLE = "Q-Learning Simulator"

black = QColor(0, 0, 0)
grey = QColor(240, 240, 240)
white = QColor(255, 255, 255)
yellow = QColor(255, 255, 0)
dark_yellow = QColor(179, 179, 0)
green = QColor(0, 255, 0)
red = QColor(255, 0, 0)


class GridWidget(QWidget):
    def __init__(self, grid, qlearning):
        super().__init__()

        self.grid = grid
        self.qlearning = qlearning

    def paintEvent(self, e):
        """
        Update the view.
        """
        qp = QPainter()
        #qp.setWindow(QRect(self.pos().x(), self.pos().y(), self.width(), self.height()))
        x_step = self.height() / self.grid.height
        y_step = self.width() / self.grid.width

        use_current = True
        action = self.qlearning.current_action

        qp.begin(self)
        pos = self.grid.current_pos if use_current else self.grid.previous_pos

        for x in range(self.grid.height):
            for y in range(self.grid.width):
                a = action if np.all(pos == [x,y]) else None
                with_pos = np.all(self.grid.current_pos == [x,y])
                if self.grid.data[x][y] == '.':
                    self.draw_empty(qp, x * x_step, y * y_step, x_step, y_step, a, with_pos)
                elif self.grid.data[x][y] == 'X':
                    qp.setBrush(grey)
                    qp.setPen(grey)
                    qp.drawRect(x* x_step, y* y_step, x_step, y_step)
                elif self.grid.data[x][y] == 'W':
                    qp.setBrush(green)
                    qp.setPen(green)
                    qp.drawRect(x* x_step, y* y_step, x_step, y_step)
                elif self.grid.data[x][y] == 'L':
                    qp.setBrush(red)
                    qp.setPen(red)
                    qp.drawRect(x* x_step, y* x_step, x_step, y_step)

        qp.end()

    def draw_empty(self, qp, x, y, x_step, y_step, action, with_pos):
        center = QPoint(x + x_step / 2, y + y_step / 2)
        top_left = QPoint(x, y)
        top_right = QPoint(x, y + y_step)
        bot_left = QPoint(x + x_step, y)
        bot_right = QPoint(x + x_step, y + y_step)

        left = (top_left, center, bot_left, Action.LEFT)
        right = (top_right, center, bot_right, Action.RIGHT)
        up = (top_left, center, top_right, Action.UP)
        down = (bot_left, center, bot_right, Action.DOWN)

        # Draw the chosen action at the end
        all_trig = [left, right, up, down]
        all_trig = sorted(all_trig, key=lambda x: 0 if x[-1] != action else 1)

        for p1, p2, p3, a in all_trig:
            if a == action:
                qp.setBrush(dark_yellow)
                qp.setPen(yellow)
            else:
                qp.setBrush(black)
                qp.setPen(grey)

            qp.drawPolygon(p1, p2, p3)

        if with_pos:
            qp.drawEllipse(center, 20, 20)


class QLearningUI(QDialog):

    def __init__(self, grid, qlearning, parent=None):
        super().__init__(parent)

        self.grid = grid
        self.qlearning = qlearning

        self.main_layout = QVBoxLayout()
        self.text = QLabel()
        self.text.setText(TEXT)
        self.main_layout.addWidget(self.text)

        self.top_layout = QHBoxLayout()
        self.grid_view = GridWidget(grid, qlearning)
        self.grid_view.setMinimumWidth(500)
        self.grid_view.setMinimumHeight(400)
        self.top_layout.addWidget(self.grid_view)

        self.main_layout.addLayout(self.top_layout)

        self.setLayout(self.main_layout)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("UI for Episode Plot")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    from pathlib import Path
    grid_path = Path('grid.txt')

    grid = Grid(grid_path)
    q_learning = QLearning(grid)

    ui = QLearningUI(grid, q_learning)
    ui.show()

    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook


    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    # noinspection PyBroadException
    try:
        sys.exit(app.exec_())
    except Exception:
        print("Exiting")
