from pathlib import Path

WEBOTS_VERSION = "R2025a"
WEBOTS_RAW_ROOT = f"https://raw.githubusercontent.com/cyberbotics/webots/{WEBOTS_VERSION}/projects"
url_1 = f"{WEBOTS_RAW_ROOT}/objects/backgrounds/protos/TexturedBackground.proto"
url_2 = f"{WEBOTS_RAW_ROOT}/objects/backgrounds/protos/TexturedBackgroundLight.proto"
url_3 = f"{WEBOTS_RAW_ROOT}/objects/floors/protos/RectangleArena.proto"
url_4 = f"{WEBOTS_RAW_ROOT}/robots/gctronic/e-puck/protos/E-puck.proto"

wbt_content = f'#VRML_SIM {WEBOTS_VERSION} utf8\nEXTERNPROTO "{url_1}"\nEXTERNPROTO "{url_2}"\nIMPORTABLE EXTERNPROTO "{url_3}"\nIMPORTABLE EXTERNPROTO "{url_4}"\n'
wbt_content += "WorldInfo {\n  basicTimeStep 20\n}\n"
Viewpoint = """
Viewpoint {
  orientation -0.5773502691896258 0.5773502691896258 0.5773502691896258 2.0944
  position 0.00011706497804167808 0.1699505320272429 4.101142536591572
}
TexturedBackground {
}
TexturedBackgroundLight {
}
"""
wbt_content += Viewpoint
num_envs = 1
num_robots = 6
receiver_template = """
    Receiver {{
      name "receiver{name}"
      channel {channel}
    }}
"""
emitter_template = """
    Emitter {{
      name "emitter{name}"
      channel {channel}
    }}
"""

receivers_emitters = ""

for i in range(num_envs):
    receivers_emitters += receiver_template.format(name=i+1,channel=1+i*(num_robots+3))
    receivers_emitters += emitter_template.format(name=i+1, channel=2+i*(num_robots+3))


SUPERVISOR = """
DEF SUPERVISOR Robot {{
  children [
{receiver_and_emitter}
  ]
  name "supervisor"
  controller "<extern>"
  supervisor TRUE
}}
"""

SUPERVISOR_content = SUPERVISOR.format(receiver_and_emitter=receivers_emitters)
wbt_content += SUPERVISOR_content
output_path = Path(__file__).with_name("generated_world.wbt")
output_path.write_text(wbt_content, encoding="utf-8")

print(f"{output_path} has been generated")
