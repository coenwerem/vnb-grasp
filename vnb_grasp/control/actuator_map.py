class ActuatorMap:
    """
    Constructs a role-aware object-to-list map of a MuJoCo model's actuators.

    NOTE: This class assumes a well-defined and preferably contiguous XML robot description
    comprising arm and hand actuators. It also supports a free-moving hand, though in that
    case the algorithm converges more slowly to a stable grasp because the hand has to do
    more work to stabilize.
    """
    def __init__(self, model):
        self.arm = []
        self.hand = []
        self.thumb = []
        self.index = []
        self.middle = []
        self.ring = []
        self.pinky = []

        for i in range(model.nu):
            name = model.actuator(i).name.lower()

            if any(k in name for k in ["shoulder", "elbow", "wrist"]):
                self.arm.append(i)

            elif "thumb" in name:
                self.thumb.append(i)
                self.hand.append(i)

            elif "index" in name:
                self.index.append(i)
                self.hand.append(i)

            elif "middle" in name:
                self.middle.append(i)
                self.hand.append(i)

            elif "ring" in name:
                self.ring.append(i)
                self.hand.append(i)

            elif "pinky" in name:
                self.pinky.append(i)
                self.hand.append(i)

        self.arm.sort()
        self.hand.sort()
