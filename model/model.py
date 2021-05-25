import os, torch, nltk
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from slp.util.embeddings import EmbeddingsLoader
from slp.helpers import PackSequence, PadPackedSequence
from utils.attention_heatmaps import generate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class ThreeEncoders(BaseModel):
    def __init__(self, input_size, hidden, batch_size, num_layers, output_size, batch_first=False, bidirectional=False, attention=True):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.hidden = hidden
        self.num_classes = output_size
        self.attention = attention

        self.diction = self.load_embeddings().to(DEVICE)
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction).to(DEVICE)
        self.pos_tags = []

        self.speaker1 = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True, bidirectional=False)
        self.speaker2 = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True, bidirectional=False)
        self.context_encoder = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=False, bidirectional=False)

        self.bidirectional = False
        hidden = 2 * self.hidden if self.bidirectional else self.hidden
        self.linear1 = nn.Linear(hidden, self.num_classes)
        self.linear2 = nn.Linear(hidden, self.num_classes)
        if self.attention:
            self.word1 = nn.Linear(hidden, hidden)
            self.context1 = nn.Linear(hidden, 1, bias=False)
            self.word2 = nn.Linear(hidden, hidden)
            self.context2 = nn.Linear(hidden, 1, bias=False)

    def load_embeddings(self):
        cwd = os.getcwd()
        loader = EmbeddingsLoader(cwd + '/data/embeddings/glove.6B.300d.txt', 300)
        word2idx, self.idx2word, embeddings = loader.load()
        embeddings = torch.tensor(embeddings)
        return embeddings.to(DEVICE)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.speaker1_hidden_state = torch.zeros(batch_size, 1, self.hidden).to(DEVICE)
        self.speaker2_hidden_state = torch.zeros(batch_size, 1, self.hidden).to(DEVICE)
        self.context_hidden_state = torch.zeros(batch_size, 1, self.hidden).to(DEVICE)
        self.context_output = torch.zeros(1, batch_size, self.hidden).to(DEVICE)

    def forward(self, utterances, speakers, heatmap=False, postags=False):
        outputs = None if self.attention else []
        i = 0
        if speakers.shape[0] == 1:
            speakers = speakers[0]
        for (utterance, speaker) in zip(utterances, speakers):
            i += 1
            output_emb = self.lookup(utterance).to(DEVICE) # (B, S, 300)
            if speaker == 0:
                inputt = torch.cat((self.context_output, output_emb), dim=1) #(B, S+1, 300)
                output, self.speaker1_hidden_state = self.speaker1(inputt, self.speaker1_hidden_state)
  
                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker1_hidden_state,
                                                                    self.context_hidden_state)
                if self.attention:
                    output_att = self.word1(output)
                    output_att = self.context1(output_att)
                    output_att = F.softmax(output_att,dim=1)
                    output = (output * output_att).sum(1)
                    if postags:
                        # keep pos tag of the highest-ranked word
                        higher_weight = torch.argmax(output_att.squeeze(2).squeeze(0)).item()
                        if higher_weight == 0:
                            higher_word = '[context]'
                        else:
                            higher_word = self.idx2word[utterance[0][higher_weight-1].item()]
                        self.pos_tags.append(nltk.pos_tag([higher_word])[0][1])
                    if heatmap:
                        import pdb; pdb.set_trace()
                        words_list = [self.idx2word[w.item()] for w in utterance[0]]
                        words_list.insert(0, '[context]')
                        att_rescaled = output_att * 100
                        attention_list = (att_rescaled.squeeze(2).squeeze(0).detach().cpu()).tolist()
                        generate(words_list, attention_list, 'results_heatmap'+str(i)+'.tex')

                output = self.linear1(output)
            else:
                inputt = torch.cat((self.context_output, output_emb), dim=1)
                output, self.speaker2_hidden_state = self.speaker2(inputt, self.speaker2_hidden_state)
                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker2_hidden_state,
                                                                    self.context_hidden_state)

                if self.attention:
                    output_att = self.word2(output)
                    output_att = self.context2(output_att)
                    output_att = F.softmax(output_att,dim=1)
                    output = (output * output_att).sum(1)
                    if postags:
                        # keep pos tag of the highest-ranked word
                        higher_weight = torch.argmax(output_att.squeeze(2).squeeze(0)).item()
                        if higher_weight == 0:
                            higher_word = '[context]'
                        else:
                            higher_word = self.idx2word[utterance[0][higher_weight-1].item()]
                        self.pos_tags.append(nltk.pos_tag([higher_word])[0][1])
                    if heatmap:
                        #import pdb; pdb.set_trace()
                        words_list = [self.idx2word[w.item()] for w in utterance[0]]
                        words_list.insert(0, '[context]')
                        att_rescaled = output_att * 100
                        attention_list = (att_rescaled.squeeze(2).squeeze(0).detach().cpu()).tolist()
                        generate(words_list, attention_list, 'results_heatmap'+str(i)+'.tex')

                output = self.linear2(output)

            if self.attention:
                if outputs == None:
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output), dim=0)
            else:
                outputs.append(output[0][-1])  # output: (B, S+1, classes
        return outputs.to(DEVICE) if self.attention else torch.stack(outputs).to(DEVICE)


class GRU(nn.Module):
    def __init__(self, input_size, hidden, batch_size, num_layers,output_size, batch_first = True, attention=True):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.batch_size = batch_size
        self.attention = attention

        self.diction = self.load_embeddings()
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction)
        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden, batch_first=bool(batch_first), bidirectional=False)
        self.output = nn.Linear(hidden, output_size)
        self.pos_tags = []
        
        if self.attention:
            self.word = nn.Linear(hidden, hidden)
            self.context = nn.Linear(hidden, 1, bias=False)
            
        self._init_hidden_state()


    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.h_0 = torch.zeros(1, batch_size, self.hidden).to(DEVICE)  # changed order
        
    def load_embeddings(self):
        cwd = os.getcwd()
        loader = EmbeddingsLoader(cwd + '/data/embeddings/glove.6B.300d.txt', 300)
        word2idx, self.idx2word, embeddings = loader.load()
        embeddings = torch.tensor(embeddings)
        return embeddings.to(DEVICE) 



    def forward(self, input, heatmap=False, postags=False):
        outputs = []; i=0
        try:
            for utterance in input:
                if utterance.size(1) == 0:
                    continue
                output_emb = self.lookup(utterance) # (B, S, 30
                out, self.h_0 = self.gru(output_emb, self.h_0)

                i += 1
                if self.attention:
                    output = self.word(out)
                    att_out = self.context(output)
                    att_out = F.softmax(att_out, dim=1)
                    out = (out * att_out).sum(1)
                    """
                    if postags:
                        # keep pos tag of the highest-ranked word
                        higher_weight = torch.argmax(att_out.squeeze(2).squeeze(0)).item()
                        higher_word = self.idx2word[utterance[0][higher_weight].item()]
                        self.pos_tags.append(nltk.pos_tag([higher_word])[0][1])
                    if heatmap:
                        words_list = [self.idx2word[w.item()] for w in utterance[0]]
                        att_rescaled = att_out * 100
                        attention_list = (att_rescaled.squeeze(2).squeeze(0).detach().cpu()).tolist()
                        generate(words_list, attention_list, 'results_heatmap'+str(i)+'.tex')
                    """
                else:
                    out = out[:, -1,:]

                out = self.output(out)
                outputs.append(out)
            final = torch.stack(outputs, dim=0)
        except:
            import pdb; pdb.set_trace()
        return final.squeeze(1).to(DEVICE)


class WordAttNet(nn.Module):
    def __init__(self, hidden_size=300):
        super(WordAttNet, self).__init__()

        self.gru = nn.GRU(300, 300, bidirectional = False, batch_first=True)
        self.word = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)
       
        self.diction = self.load_embeddings()
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction)
        
    def load_embeddings(self):
        cwd = os.getcwd()
        loader = EmbeddingsLoader(cwd + '/data/embeddings/glove.6B.300d.txt', 300)
        word2idx, self.idx2word, embeddings = loader.load()
        embeddings = torch.tensor(embeddings)
        return embeddings.to(DEVICE) 

    def forward(self, inputs, hidden_state):
        #import pdb; pdb.set_trace()
        output_emb = self.lookup(inputs)
        f_output, h_output = self.gru(output_emb.float(), hidden_state)
        
        output = self.word(f_output)
        output = self.context(output)
        output = F.softmax(output, dim=1)

        output = (f_output * output).sum(1)
        return output, h_output


class SentAttNet(nn.Module):
    def __init__(self, hidden_size=300, num_classes=0):
        super(SentAttNet, self).__init__()
        num_classes = num_classes
        self.gru = nn.GRU(hidden_size, 300, bidirectional=False, batch_first=True) #changed hidden size
        self.sent = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, inputs, hidden_state):
        #import pdb; pdb.set_trace()
        outputs = []
        for sentence in inputs:
            f_output, h_output = self.gru(sentence.unsqueeze(0), hidden_state)
        
            output = self.sent(f_output)
            output = self.context(output)
            output = F.softmax(output, dim=1)
            output = (f_output * output).sum(1)
            output = self.fc(output).squeeze()
            outputs.append(output.unsqueeze(0))
        #import pdb; pdb.set_trace()
        outputs = torch.stack(outputs,dim=1)
        return outputs.squeeze(0), h_output


class HierAttNet(nn.Module):
    def __init__(self, hidden_size, batch_size, num_classes):
        super (HierAttNet, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        
        self.sent_att_net = SentAttNet(self.hidden_size, num_classes)
        self.word_att_net_text = WordAttNet(self.hidden_size)

        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
        self.sent_hidden_state = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)


    def forward(self, inputs):
        # inputs = (B, S, W)
        #import pdb; pdb.set_trace()
        output_list_text = []

        for i in inputs:
            if i.size(1) == 0:
                continue
            output_text, self.word_hidden_state = self.word_att_net_text(i, self.word_hidden_state) #[8,600]
            output_list_text.append(output_text)
#            import pdb; pdb.set_trace()
            self.word_hidden_state = repackage_hidden(self.word_hidden_state)
        
        # output_list_text = (S, B, 300)
        output, self.sent_hidden_state = self.sent_att_net(output_list_text, self.sent_hidden_state)
        self.sent_hidden_state = repackage_hidden(self.sent_hidden_state)
        return output


"""
 "arch": {
        "type": "HierAttNer",
        "args": {
            "input_size": 300,
            "hidden": 300,
            "batch_size": 1,
            "num_layers": 1,
            "output_size": 7,
            "batch_first": "True",
            "attention": "False"
        }
"""