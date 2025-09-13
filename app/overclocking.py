import torch
import time

print("heyaaa! get ready to see some serious computing action! o/\n")
print("this script will run calculations non-stop until you press Ctrl+C!\n")

# let's set some big-ish dimensions for our matrices!
# not tooooo huge that it takes ages for one pass, but big enough to be meaningful
MATRIX_SIZE = 4096 # this will create matrices of 4096x4096, which is still chunky!
NUM_ITERATIONS = 5 # just to show how many iterations it does before printing a status message

# check if we can use the GPU (which pytorch calls 'cuda' for both nvidia/amd)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("yayy! using your GPU for this continuous computation! gooo gpu! -w-")
    print(f"gpu device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("awww, no GPU detected, so we're falling back to the CPU. it'll still work!")

print(f"\npreparing to crunch {MATRIX_SIZE}x{MATRIX_SIZE} matrix multiplications on the {device.type}...")
print("watch your GPU/CPU usage now! press Ctrl+C to stop.\n")

# create two big random matrices once, to reuse in the loop
# .half() makes them use float16, which is often faster on GPUs and uses less memory!
matrix_a = torch.rand(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float16)
matrix_b = torch.rand(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float16)

iteration_count = 0
try:
    while True: # infinite loop for non-stop computing!
        # perform the super big matrix multiplication!
        # this is where your GPU/CPU will really shine!
        _ = torch.matmul(matrix_a, matrix_b)

        # for GPU, important: wait for the GPU to finish its work for this iteration
        # this makes sure the GPU is *actually* busy and not just queuing
        if device.type == 'cuda':
            torch.cuda.synchronize()

        iteration_count += 1
        if iteration_count % NUM_ITERATIONS == 0:
            print(f"did {iteration_count} big calculations! still crunching... (-w-)")

except KeyboardInterrupt:
    print("\n\nawww, you pressed Ctrl+C! stopping the computing frenzy. <3")
    print(f"total calculations done: {iteration_count}")
    print("hope you saw your GPU/CPU working hard! o/")
except Exception as e:
    print(f"\n\noopsie! something went wrong: {e}")
