import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seaborn.palettes import color_palette

baseline_pos = np.loadtxt('./baseline_POS.txt', dtype = str)
baseline_freq = np.loadtxt('./baseline_POS_frequency.txt', dtype = int)

three_enc_pos = np.loadtxt('./ThreeEncoders_POS.txt', dtype = str)
three_enc_freq = np.loadtxt('./ThreeEncoders_POS_frequency.txt', dtype = int)

pos_label_map = pd.read_csv('./pos_map.tsv', sep = '\t', index_col=[0])

baseline = pd.DataFrame({'POS': baseline_pos, 'frequency': baseline_freq})
three_enc = pd.DataFrame({'POS': three_enc_pos, 'frequency': three_enc_freq})

#select the top 10 labels as the rest is not as informative
baseline_top10 = baseline.sort_values(by = ['frequency'], ascending= False)[:10]
three_enc_top10 = three_enc.sort_values(by = ['frequency'], ascending= False)[:10]

#plot either model separately
models = [baseline_top10, three_enc_top10]
model_names = ['Baseline GRU', 'ThreeEncoders']

fig, axs = plt.subplots(1,2, figsize = (15, 5))

for i, ax in enumerate(axs.flatten()):
    sns.barplot(data = models[i], x = 'POS', y = 'frequency', ax = ax, palette = 'magma')
    ax.set_xticklabels(pos_label_map.loc[models[i]['POS']]['explanation'], rotation = 45, ha="right")
    ax.set_xlabel('Part of Speech')
    ax.set_ylabel('Count')
    ax.set_title(model_names[i])

plt.suptitle('10 most frequent Word Classes per Model')   
plt.tight_layout()
plt.savefig('./POS_frequency_per_model.pdf')
plt.savefig('./POS_frequency_per_model.png')
plt.close()


#plot both in the same plot
baseline_top10['model'] = 'Baseline GRU'
three_enc_top10['model'] = 'Three Encoders'

both = pd.concat([baseline_top10, three_enc_top10])

plot = sns.barplot(data = both, x = 'POS', y = 'frequency', hue = 'model', palette = 'magma')
label_tags = [label.get_text() for label in plot.get_xticklabels()]
plot.set_xticklabels(pos_label_map.loc[label_tags]['explanation'], rotation = 45, ha="right")
plot.set_xlabel('Part of Speech')
plot.set_ylabel('Count')
plot.set_title('10 most frequent Word Classes per Model')

plt.tight_layout()
plt.savefig('./POS_frequency_both_models.pdf')
plt.savefig('./POS_frequency_both_models.png')
plt.close()


#plot word classes with largest differences between the models

unique_tags = baseline['POS'].append(three_enc['POS']).unique()
diff_pos = {}

for tag in unique_tags:
    freq1 = baseline.loc[baseline['POS'] == tag]['frequency'].values
    freq2 = three_enc.loc[three_enc['POS'] == tag]['frequency'].values

    if freq1.size == 0:
        freq1 = 0
    if freq2.size == 0:
        freq2 = 0

    diff = freq1 - freq2
    
    diff_pos[tag] = abs(diff)[0]


diff_plot = sns.barplot(data = pd.DataFrame(diff_pos.items(), columns=['POS', 'frequency']).sort_values(by = ['frequency'], ascending= False)[:10],
                        x = 'POS', y = 'frequency', palette = 'magma')

label_tags = [label.get_text() for label in diff_plot.get_xticklabels()]
diff_plot.set_xticklabels(pos_label_map.loc[label_tags]['explanation'], rotation = 45, ha="right")
diff_plot.set_xlabel('Part of Speech')
diff_plot.set_ylabel('Absolute Difference Between Models')
diff_plot.set_title('10 highest Differences in Word Class Frequencies between Models')
plt.tight_layout()
plt.savefig('./POS_frequency_difference.pdf')
plt.savefig('./POS_frequency_difference.png')
plt.close()

