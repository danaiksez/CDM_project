# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
# import data_loader.data_loaders as module_data
# import model.loss as module_loss
# import model.metric as module_metric
# import model.model as module_arch
# from parse_config import ConfigParser
# from utils import inf_loop, MetricTracker
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.ticker as ticker

## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
	word_num = len(text_list)
	text_list = clean_word(text_list)
	# for idx in range(word_num):
	print(attention_list)
	print(text_list)
	print('len',len(attention_list))
	evaluateAndShowAttention(text_list, [1 for i in range(len(attention_list))],attention_list,1 )
# 	with open(latex_file,'w') as f:
# 		f.write(r'''\documentclass[varwidth]{standalone}
# \special{papersize=210mm,297mm}
# \usepackage{color}
# \usepackage{tcolorbox}
# \usepackage{CJK}
# \usepackage{adjustbox}
# \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
# \begin{document}
# \begin{CJK*}{UTF8}{gbsn}'''+'\n')
# 		string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
# 		for idx in range(word_num):
# 			string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
# 		string += "\n}}}"
# 		f.write(string+'\n')
# 		f.write(r'''\end{CJK*}
# \end{document}''')
def showAttention(input_sentence, output_words, attentions,i):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.reshape(np.array(attentions),(1,len(attentions))), cmap='bone')
    fig.colorbar(cax)
    # Set up axes

    print(len(input_sentence),len(output_words),len(attentions))
    ax.set_xticklabels(input_sentence, rotation=90)
    ax.set_yticklabels(output_words, rotation=90)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence, output, attention,i):
    print('input =', input_sentence)
    print(output)
    print('output =', output)
    print('index',i)
    showAttention(input_sentence, output, attention,i)

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


if __name__ == '__main__':
	## This is a demo:

	sent = '''the USS Ronald Reagan - an aircraft carrier docked in Japan - during his tour of the region, vowing to "defeat any attack and meet any use of conventional or nuclear weapons with an overwhelming and effective American response".
North Korea and the US have ratcheted up tensions in recent weeks and the movement of the strike group had raised the question of a pre-emptive strike by the US.
On Wednesday, Mr Pence described the country as the "most dangerous and urgent threat to peace and security" in the Asia-Pacific.'''
	words = sent.split()
	word_num = len(words)
	attention = [(x+1.)/word_num*100 for x in range(word_num)]
	import random
	random.seed(42)
	random.shuffle(attention)
	color = 'red'
	generate(words, attention, "sample.tex", color)
