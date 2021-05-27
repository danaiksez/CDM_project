import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import inf_loop, MetricTracker
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.ticker as ticker
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_prec_rec_f1(output, target):
    with torch.no_grad():
        pred = F.softmax(output, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.clone().cpu().numpy()
        labs = target.clone().cpu().numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(labs, pred, labels=[0,1,2,3,4,5,6], zero_division=0)
    return prec, rec, f1

def _progress(batch_idx,len_epoch, test_data_loader):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(test_data_loader, 'n_samples'):
        current = batch_idx * test_data_loader.batch_size
        total = test_data_loader.n_samples
    else:
        current = batch_idx
        total = len_epoch
    return base.format(current, total, 100.0 * current / total)


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

def generate(text_list, attention_list, sent1,labels, i,color='red', rescale_value = False):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
		# print(len(text_list), text_list)
	word_num = len(text_list)
	# text_list = clean_word(text_list)
	output = []
	evaluateAndShowAttention(sent1,labels,attention_list,i )

def showAttention(input_sentence, output_words, attentions,i):
    # Set up figure with colorbar
	fig = plt.figure()
	index = 0
	# fig = plt.figure()
	gs = fig.add_gridspec(len(input_sentence), hspace=3)
	ax = gs.subplots(sharex=False, sharey=False)
	# fig.suptitle('attention heatmap')
# Hide x labels and tick labels for all but bottom plot.
	print('attentions', len(attentions), attentions)
	for seq in range(len(input_sentence)):
		input0 = clean_word(input_sentence[seq][0].split(' '))
		input = [x for x in input0 if x]
		print(input)
		output = [output_words[0][0] for i in range(len(input))]

		# print('output',output)
		attention = attentions[seq]
		print(len(input),len(output), len(attention))
		cax = ax[seq].matshow(np.reshape(np.array(attention),(1,len(attention))), cmap='bone')
		fig.colorbar(cax)
		ax[seq].set_xticklabels([" "] + input, rotation=90)
		ax[seq].set_yticklabels(output)

		# Show label at every tick
		ax[seq].xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax[seq].yaxis.set_major_locator(ticker.MultipleLocator(1))
		index += len(input)
	plt.savefig('heatmap/{}{}.jpg'.format(emotion, i))


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


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['test_data_loader']['type'])(
        config['test_data_loader']['args']['data_dir'],
        split = 'test',
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns])
    test_data_loader = config.init_obj('train_data_loader', module_data)
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    criterion = getattr(module_loss, config['loss'])
    len_epoch = len(test_data_loader)
    with torch.no_grad():
        # for batch_idx, (data, target, actions, speakers, input_sen, labels) in enumerate(test_data_loader):
        dataloader = test_data_loader.dataset.emotion_dataset
        for batch_idx, (data, speakers, input_sen, labels) in enumerate(dataloader[0]):
            # print(data, 'speaker',speakers, target)
            # data, spkrs = data.to(DEVICE), spkrs.to(DEVICE)
            target = target.to(DEVICE)
            # print('tsrget', target[0])
            model._init_hidden_state()

            output, text_list, attention = model(data, speakers, heatmap=True, postags=False)
            generate(text_list, attention,input_sen, labels,('no_emotion',batch_idx))
            if output.size(0) != target.size(1):
                if output.size(0) < target.size(1):
                    idx = []
                    for i in range(len(data)):
                        if data[i].size(1) == 0:
                            idx.append(i)
                    target_numpy = target.clone().cpu()
                    target_numpy = np.delete(target_numpy, idx)
                    target = torch.tensor(target_numpy).to(DEVICE)
                    # print("newtarget",target)
                elif output.size(0) > target.size(1):
                    output = output[:target.size(1)]
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = len(data)
            total_loss += loss.item() * batch_size
            for met in metric_fns:
                if met.__name__ == 'accuracy':
                    test_metrics.update(met.__name__, met(output, target.squeeze(0)))
                else:
                    classid = int(met.__name__[-1])  # id of the class (number)
                    pred = torch.argmax(F.softmax(output, dim=1), dim=1)
                    if classid in pred:     # update metric score only if class is present in the targets
                        scores = compute_prec_rec_f1(output, target.squeeze(0))
                        test_metrics.update(met.__name__, met(scores))

            if batch_idx % 100 == 0:
                logger.debug('Test Epoch: {} {} Loss: {:.6f}'.format(1,
                    _progress(batch_idx, len_epoch, test_data_loader),
                    loss.item()))

            if batch_idx == len_epoch:
                break
        log = test_metrics.result()

        # if lr_scheduler is not None:
        #     lr_scheduler.step()

        #import pdb; pdb.set_trace()
        print(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
