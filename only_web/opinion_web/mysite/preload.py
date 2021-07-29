from only_web.mining.triple_model  import get_opt
from only_web.MyOTE.models.ote import OTE
from only_web.MyOTE.data_utils import ABSADataReader,build_tokenizer,build_embedding_matrix
import time
import os
import torch
from transformers import BertTokenizer


def load_model():
    start_time = time.time()
    opt = get_opt()

    #bert
    tokenizer_bert = BertTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert-base-chinese'))
    embedding_matrix_bert = torch.tensor([0])
    #LSTM
    tokenizer_LSTM = build_tokenizer(data_dir=opt.data_dir)
    embedding_matrix_LSTM = build_embedding_matrix(opt.data_dir, tokenizer_LSTM.word2idx, opt.embed_dim, opt.dataset)

    absa_data_reader = ABSADataReader(data_dir=opt.data_dir,opt=opt)
    idx2tag,idx2polarity,idx2target,id2express =  absa_data_reader.reverse_tag_map, \
                                                  absa_data_reader.reverse_polarity_map, \
                                                  absa_data_reader.reverse_target_map,\
                                                  absa_data_reader.reverse_express_map


    model_bert = OTE(embedding_matrix=embedding_matrix_bert,
                opt=opt,
                idx2tag=idx2tag,
                idx2polarity=idx2polarity,
                idx2target = idx2target,
                idx2express =id2express,
                ).to(opt.device)

    model_bert.load_state_dict(torch.load(opt.state_dict_path['ote_Bert'], map_location=lambda storage, loc: storage))
    model_bert.eval()

    model_LSTM = OTE(embedding_matrix=embedding_matrix_LSTM,
                     opt=opt,
                     idx2tag=idx2tag,
                     idx2polarity=idx2polarity,
                     idx2target = idx2target,
                     idx2express =id2express,
                     ).to(opt.device)

    model_LSTM.load_state_dict(torch.load(opt.state_dict_path['ote_LSTM'], map_location=lambda storage, loc: storage))
    model_LSTM.eval()
    return tokenizer_bert, tokenizer_LSTM, model_bert, model_LSTM