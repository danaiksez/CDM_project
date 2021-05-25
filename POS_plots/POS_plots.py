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

#plot eithe rmodel separately
models = [baseline_top10, three_enc_top10]
model_names = ['Baseline GRU', 'ThreeEncoders']

fig, axs = plt.subplots(1,2, figsize = (15, 5))

for i, ax in enumerate(axs.flatten()):
    sns.barplot(data = models[i], x = 'POS', y = 'frequency', ax = ax, palette = 'magma')
    ax.set_xticklabels(pos_label_map.loc[models[i]['POS']]['explanation'], rotation = 45, ha="right")
    ax.set_xlabel('Part of Speech')
    ax.set_ylabel('Count')
    ax.set_title(model_names[i])
    
plt.tight_layout()
plt.savefig('./POS_frequency_per_model.pdf')
plt.savefig('./POS_frequency_per_model.png')


#plot both in the same plot
baseline_top10['model'] = 'Baseline GRU'
three_enc_top10['model'] = 'Three Encoders'

both = pd.concat([baseline_top10, three_enc_top10])

plot = sns.barplot(data = both, x = 'POS', y = 'frequency', hue = 'model')
label_tags = [label.get_text() for label in plot.get_xticklabels()]
plot.set_xticklabels(pos_label_map.loc[label_tags]['explanation'], rotation = 45, ha="right")
plot.set_xlabel('Part of Speech')
plot.set_ylabel('Count')
plt.tight_layout()
plt.savefig('./POS_frequency_both_models.pdf')
plt.savefig('./POS_frequency_both_models.png')


