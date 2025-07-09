
import csv
import matplotlib.pyplot as plt

data = {
    'duration': [],
    '0': [],
    '1': []
}
total_shots = 1000  


with open('rabi_data.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data['duration'].append(float(row['duration']))
        data['0'].append(int(row['count0'])/total_shots)
        data['1'].append(int(row['count1'])/total_shots)


plt.figure(figsize=(10,6))
plt.plot(data['duration'], data['0'], 'b-o', label='State |0>')
plt.plot(data['duration'], data['1'], 'r--s', label='State |1>')


plt.title('Rabi Oscillation Experiment')
plt.xlabel('Duration (dt)')
plt.ylabel('Probability')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()


plt.savefig('rabi.png', dpi=300)
plt.show()