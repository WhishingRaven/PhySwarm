# generate_wbt.py

url_1 = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
url_2 = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
url_3 = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"
url_4 = "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/gctronic/e-puck/protos/E-puck.proto"

wbt_content = f'#VRML_SIM R2023b utf8\nEXTERNPROTO "{url_1}"\nEXTERNPROTO "{url_2}"\nIMPORTABLE EXTERNPROTO "{url_3}"\nIMPORTABLE EXTERNPROTO "{url_4}"\n'
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
with open("generated_world.wbt", "w") as f:
    f.write(wbt_content)

print("generated_world.wbt has been generated")
