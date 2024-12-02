from transitions import Machine


class RobotFSM:
    # Define states
    states = ["idle", "waiting", "move_by_sampling", "move_by_graph", "error"]

    def __init__(self, name: str = "Unnamed_robot") -> None:

        self.name = name
        self.debug = False  # debug messages

        self.machine = Machine(model=self, states=RobotFSM.states, initial="idle")

        # Define transitions
        self.machine.add_transition(trigger="lose_control", source="*", dest="error")
        self.machine.add_transition(
            trigger="restore_control", source="error", dest="idle"
        )
        self.machine.add_transition(
            trigger="startup",
            source="idle",
            dest="waiting",
            before="startup_transition",
        )
        self.machine.add_transition(
            trigger="wait", source=["move_by_sampling", "move_by_graph"], dest="waiting"
        )
        self.machine.add_transition(
            trigger="resume_sampling", source="waiting", dest="move_by_sampling"
        )
        self.machine.add_transition(
            trigger="resume_graph", source="waiting", dest="move_by_graph"
        )

        self.machine.on_enter_error(self.on_enter_error)

    def on_enter_error(self):
        if self.debug:
            print("Entering Error: Robot control lost, please fix", flush=True)

    def startup_transition(self):
        if self.debug:
            print(f"Robot {self.name} is waiting and ready to move")

    def report_status(self):
        print(f"Current state: {self.state}")


if __name__ == "__main__":
    robot = RobotFSM("test_rob")
    robot.report_status()

    robot.startup()
