"""Dependency-light CSV messaging base for Webots robot controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from controller import Robot


class CSVRobot(Robot, ABC):
    """Exchange list-like messages through a Webots emitter and receiver."""

    def __init__(
        self,
        timestep: int | None = None,
        emitter_name: str = "emitter",
        receiver_name: str = "receiver",
    ):
        Robot.__init__(self)
        self.timestep = int(timestep or self.getBasicTimeStep())
        self.emitter = self.getDevice(emitter_name)
        self.receiver = self.getDevice(receiver_name)
        self.receiver.enable(self.timestep)

    def handle_emitter(self) -> None:
        message = self.create_message()
        if isinstance(message, str):
            payload = message
        elif isinstance(message, Sequence):
            payload = ",".join(map(str, message))
        else:
            raise TypeError("create_message() must return a string or sequence")
        self.emitter.send(payload.encode("utf-8"))

    def handle_receiver(self) -> None:
        if self.receiver.getQueueLength() == 0:
            return
        try:
            message = self.receiver.getString()
        except AttributeError:
            message = self.receiver.getData().decode("utf-8")
        self.receiver.nextPacket()
        self.use_message_data(message.split(","))

    @abstractmethod
    def create_message(self) -> str | Sequence[object]:
        """Build the next outbound CSV message."""

    @abstractmethod
    def use_message_data(self, message: list[str]) -> None:
        """Apply a decoded inbound CSV message."""
