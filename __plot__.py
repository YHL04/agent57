import pandas as pd
import os
import matplotlib.pyplot as plt

dir = os.listdir("logs")
dir.sort()
latestfile = "logs/" + dir[-1]

data = pd.read_csv(latestfile,
                   names=["Time",
                          "Updates",
                          "Frames",
                          "Loss",
                          "Reward",
                          "Epsilon"]
                   )

plt.subplot(5, 1, 1)
plt.title("Time (Sec)")
plt.plot(data["Time"])

plt.subplot(5, 1, 2)
plt.title("Updates")
plt.plot(data["Updates"])

plt.subplot(5, 1, 3)
plt.title("Frames")
plt.plot(data["Frames"])

plt.subplot(5, 1, 4)
plt.title("Loss (Log scale)")
plt.yscale("log")
plt.plot(data["Loss"])

plt.subplot(5, 1, 5)
plt.title("Normalized Reward")
plt.plot(data["Reward"])
plt.plot(data["Reward"].rolling(200).mean())

plt.show()
