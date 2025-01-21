import numpy as np
import matplotlib.pyplot as plt
# See walk.py for the implementation
import walk


n_stories = 1000
t_max = 200
positions = np.empty((n_stories, t_max))

for i_story in range(n_stories):
    positions[i_story, :] = walk.walk(t_max, 1)

# Determine the time evolution of the root-mean-square distance.
sq_distance = positions**2
# Root mean square distance
rms_distance = np.sqrt(np.mean(sq_distance, axis=0))

t = np.arange(t_max)

fig, ax = plt.subplots()
ax.plot(t, rms_distance, "g", label="ensemble RMS distance")
ax.plot(t, np.sqrt(t), "k--", label=r"theoretical $\sqrt{\langle (\delta x)^2 \rangle}$")
ax.set(xlabel="time", ylabel="x")
ax.legend()

plt.tight_layout()
plt.show()
