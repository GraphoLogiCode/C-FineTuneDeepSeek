import openvino as ov
core = ov.Core()
devices = core.available_devices
print("Available devices:", devices)

if 'NPU' in devices:
    print("NPU is ready!")
else:
    print("NPU not detected—check drivers or WSL2 compatibility.")