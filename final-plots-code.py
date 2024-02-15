import matplotlib.pyplot as plt

df = pd.read_csv("hippo-final-data.csv")

# Extracting the data for plotting
x1 = df['Sequentiality_FullSize']
x2 = df['fw_score_normalized']
x3 = df['Topic_NLL_FullSize']
x4 = df['avg_sentence_length']
y = df['memType']

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Scatter plot for each subplot
axs[0, 0].scatter(x1, y, color='red')
axs[0, 0].set_title('Sequentiality vs MemType')
axs[0, 0].set_xlabel('Sequentiality_FullSize')
axs[0, 0].set_ylabel('MemType')

axs[0, 1].scatter(x2, y, color='blue')
axs[0, 1].set_title('FW Score vs MemType')
axs[0, 1].set_xlabel('fw_score_normalized')
axs[0, 1].set_ylabel('MemType')

axs[1, 0].scatter(x3, y, color='green')
axs[1, 0].set_title('Topic NLL vs MemType')
axs[1, 0].set_xlabel('Topic_NLL_FullSize')
axs[1, 0].set_ylabel('MemType')

axs[1, 1].scatter(x4, y, color='purple')
axs[1, 1].set_title('Average Sentence Length vs MemType')
axs[1, 1].set_xlabel('avg_sentence_length')
axs[1, 1].set_ylabel('MemType')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


########################################
# Plotting the accuracy values for different models

import matplotlib.pyplot as plt

res_accuracy = [['LSTM', 0.6964769647696477],  ['Random Forest', 0.6982836495031617], ['SVC', 0.6991869918699187], ['GRU', 0.7064137308039747],['Logistic Regression', 0.7082204155374887], ['XGBoost', 0.7235772357723578]]

# Splitting data into x and y coordinates
x, y = zip(*res_accuracy)

# Creating the plot
plt.plot(x, y, marker='o')

plt.title('Accuracy plot for different models')
plt.xlabel('Model Name')
plt.ylabel('Accuracy')

# Rotating x-axis labels by 20 degrees
plt.xticks(rotation=20)

# Adjusting the bottom margin to ensure labels are not cut off
plt.subplots_adjust(bottom=0.15)

# Show the plot
plt.show()
