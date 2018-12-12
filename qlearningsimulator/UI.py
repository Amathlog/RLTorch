from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QLabel, QDialog


TEXT = "In this simple 4x3 grid-world, Q-learning agent learns by trial and error from interactions with the \n" \
       "environment. Agent starts the episode in the bottom left corner of the maze. The goal of the agent is to \n" \
       "maximize its total (future) reward. It does this by learning which action is optimal for each state. The \n" \
       "action that is optimal for each state is the action that has the highest long-term reward. Episode \n" \
       "terminates when the agent reaches +1 or -1 state, in all other states agent, will receive an immediate \n" \
       "reward -0.1. If the agent enters the wall, it bounces back. At the beginning of each game, all Q values \n" \
       "are set to zero."

WINDOW_TITLE = "Q-Learning Simulator"


class QLearningUI(QDialog):

    def __init__(self, grid, parent=None):
        super().__init__(parent)

        self.grid = grid

        self.main_layout = QVBoxLayout()
        self.text = QLabel()
        self.text.setText(TEXT)
        self.main_layout.addWidget(self.text)

        self.top_layout = QHBoxLayout()


        self.main_layout.addLayout(self.top_layout)

        self.setLayout(self.main_layout)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("UI for Episode Plot")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    ui = QLearningUI(None)
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

    try:
        sys.exit(app.exec_())
    except Exception:
        print("Exiting")
