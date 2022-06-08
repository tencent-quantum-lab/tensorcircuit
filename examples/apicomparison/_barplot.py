import numpy as np
from matplotlib import pyplot as plt

labels = ["VQE subtask", "QML subtask"]
tfqlines = [47, 32]
pllines = [29, 18]
tclines = [20, 16]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects0 = ax.bar(x - width, tfqlines, width, label="tfq")
rects1 = ax.bar(x, pllines, width, label="pennylane")
rects2 = ax.bar(x + width, tclines, width, label="tc (ours)")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Lines of code")
ax.set_title("API comparison")
ax.set_xticks(x, labels)
ax.legend()
ax.set_ylim(0, 51)
ax.set_yticks([0, 10, 20, 30, 40])
ax.bar_label(rects0, padding=1)
ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=1)

fig.tight_layout()

# plt.show()
plt.savefig("apic.pdf")
