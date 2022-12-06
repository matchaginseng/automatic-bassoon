import matplotlib.pyplot as plt

files = [
    "lr_0.0001_adam.txt", "lr_0.001_adam.txt", "lr_0.01_adam.txt", 
    "lr_0.01_sgd.txt", "lr_0.1_sgd.txt",
    "dr_0.3.txt", "dr_0.5.txt", "dr_0.7.txt", "dr_0.9.txt",
    "adadelta.txt"]

ACC = "Accuracy: "
TIME = ": time="
ENERGY = ", energy="
THRESHOLD = 0.5

energy_time_05 = []
recorded = set()
for file in files:
    accuracies = []
    times = []
    energy_consumption = []
    with open(file) as f:
        line = f.readline()
        accuracy = 0
        while line:
            if "Validation Epoch:" in line:
                accuracy = float(line[line.index(ACC) + len(ACC): line.index("\n")])
                accuracies.append(accuracy)
            elif "Up to epoch" in line:
                if "epoch 0" not in line:
                    time = float(line[line.index(TIME) + len(TIME): line.index(ENERGY)])
                    energy = float(line[line.index(", energy=") + len(", energy="): line.index(", cost=")])
                    times.append(time)
                    energy_consumption.append(energy)
                if accuracy >= THRESHOLD and file not in recorded:
                    energy_time_05.append((energy, time, file))
                    recorded.add(file)
            line = f.readline()

    plt.plot(energy_consumption, accuracies, label=file)

plt.xlabel("Energy (J)")
plt.ylabel("Accuracy")
plt.title("Energy vs Accuracy")
plt.legend()
plt.show()

energies = [elem[0] for elem in energy_time_05]
times = [elem[1] for elem in energy_time_05]
names = [elem[2] for elem in energy_time_05]

print(energy_time_05)
for i in range(len(times)):
    plt.scatter(energies[i], times[i], label=names[i])

plt.xlabel("Energy (J)")
plt.ylabel("Time (s)")
plt.title(f"Energy vs Time to Reach Accuracy of {THRESHOLD}")
plt.legend()
plt.show()
