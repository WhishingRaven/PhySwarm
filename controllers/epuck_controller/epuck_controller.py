# coding = utf-8
from pathlib import Path
import sys

CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]
if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_ROOT))

from common.webots_runtime import configure_webots_runtime

configure_webots_runtime()

from common.csv_robot import CSVRobot
from Swarm_Foraging.supervisor_controller.config import get_config
args = get_config().parse_known_args()[0]

class Epuck2Robot(CSVRobot):
    def __init__(self):
        super().__init__(timestep=args.timestep)
        self.interval = args.interval
        self.max_speed = 6.28
        self.robot_name = self.getName()[-1]
        self.timestep = args.timestep
        self.num_agents = args.num_agents

        self.ps_sensor = []
        for i in range(8):
            self.ps_sensor.append(self.getDevice(f"ps{i}"))
            self.ps_sensor[i].enable(self.timestep)

        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.timestep//self.interval)
        self.accelerometer = self.getDevice("accelerometer")
        self.accelerometer.enable(self.timestep//self.interval)

        self.wheels = []
        for wheel_name in ['left wheel motor', 'right wheel motor']:
            wheel = self.getDevice(wheel_name)  
            wheel.setPosition(float('inf')) 
            wheel.setVelocity(0.0)  
            self.wheels.append(wheel)

    def run(self):
        i = 0
        while self.step(self.timestep//self.interval) != -1:
            if i%self.interval == 0:
                self.handle_receiver()
                i = 0

            self.handle_emitter()
            i += 1

    def create_message(self):
        message = []
        message.append('a'+self.getName()[-1])
        message.extend(self.gyro.getValues())
        message.extend(self.accelerometer.getValues())
        message.extend([self.wheels[0].getVelocity()])
        message.extend([self.wheels[1].getVelocity()])
        for rangefinder in self.ps_sensor:
            message.append(rangefinder.getValue())
        return message

    def use_message_data(self, message):
        name_parts = self.getName().split('-') 
        number_str = name_parts[-1]      
        idx = int(number_str) - 1

        if idx * 2 + 1 >= len(message):
            print(f"Error: Agent {self.getName()} trying to access index {idx} but message len is {len(message)}")
            return

        speed = [float(message[idx * 2]), float(message[idx * 2 + 1])]

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(speed[i])

robot_controller = Epuck2Robot()
robot_controller.run()
