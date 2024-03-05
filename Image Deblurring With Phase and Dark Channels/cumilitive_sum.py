import matplotlib.pyplot as plt
import numpy as np

# Your actual datasets
dataset1 = np.array([0.3860871792, 0.3739714921, 0.3444338739, 0.4482295811, 0.718277514,
                    0.1798544675, 0.365326643, 0.3384941816, 0.3136948347, 0.3756439984,
                    0.3917750418, 0.4040096104, 0.6431339383, 0.1961483806, 0.4867587984,
                    0.1773705781, 0.430596292, 0.3933272958, 0.2929853499, 0.1722365022,
                    1.026009083, 0.1339962035, 0.212149635, 0.1386235356, 0.270808816,
                    0.4703245461, 0.4472700357, 0.3838211894, 0.9034196138, 0.1685884893,
                    0.9833256006, 0.154643327])

dataset2 = np.array([0.7905114889,
0.6006788015,
1.117457747,
1.211222291,
1.179512739,
0.2067798972,
0.7146824002,
0.5870229006,
0.5096021891,
0.3482043445,
2.405425787,
1.887904644,
2.009513855,
0.5696044564,
2.709815025,
1.035262942,
0.9106037617,
2.034498692,
0.3597659171,
3.15245676,
1.040705681,
1.859998107,
0.3538458049,
1.203986168,
0.557433188,
0.2928296626,
0.04722134024,
0.7529116273,
3.037003756,
0.5600508451,
1.952664375,
0.2676841617
])

# Calculate the cumulative distribution for each dataset
sorted_data1 = np.sort(dataset1)
cumulative_percentage1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1) * 100

sorted_data2 = np.sort(dataset2)
cumulative_percentage2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2) * 100

# Plot the cumulative distribution for each dataset
plt.plot(np.concatenate([[0], sorted_data1, [5]]), np.concatenate([[0], cumulative_percentage1, [100]]), label='Dataset 1')
plt.plot(np.concatenate([[0], sorted_data2, [5]]), np.concatenate([[0], cumulative_percentage2, [100]]), label='Dataset 2')

# Set axis labels and title
plt.xlabel('X')
plt.ylabel('Percentage of Data Below X')
plt.title('Cumulative Distribution Plot')

# Set X-axis limits based on your requirement (0 to 5)
plt.xlim(0, 5)

# Show the legend
plt.legend()

# Show the plot
plt.show()
