from only_web.mining.triple_model  import tripleModel,get_opt
from only_web.mining import Vocab
from only_web.mining.crawler import *
from only_web.mining.unit import *
from only_web.MyOTE.models.ote import OTE
from only_web.MyOTE.data_utils import ABSADataReader,build_tokenizer,build_embedding_matrix
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import json
import threading
import torch
from transformers import BertTokenizer
import numpy as np
from selenium import webdriver
from django.conf import settings


# triple_info = [{'text': '我是中国人','aspect':['我','中国人'],'opinion':['我','是中国人'],'triples':[('我','33','POS'),('你','吗','POS')],'sen_polarity':'POS'},
#                {'text': '我是中国人','aspect':['我'],'opinion':['是中国人'],'triples':[('他','22','POS')],'sen_polarity':'POS'}]
#
# target_info = [[('我','24','国籍','人种'),('你','24','国籍','人种')],[('他','45','a','人asfa')]]


np.seterr(divide='ignore', invalid='ignore')
# triple_info = [{'text': '我是中国人','aspect':['我','中国人'],'opinion':['我','是中国人'],'triples':[('我','33','POS'),('你','吗','POS')],'sen_polarity':'POS'},
#                {'text': '我是中国人','aspect':['我'],'opinion':['是中国人'],'triples':[('他','22','POS')],'sen_polarity':'POS'}]
#
# target_info = [[('我','24','国籍','人种'),('你','24','国籍','人种')],[('他','45','a','人asfa')]]


def load_model():
    global tokenizer_bert, tokenizer_LSTM, model_bert, model_LSTM
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
    # model_LSTM,tokenizer_LSTM = [],[]
    return tokenizer_bert, tokenizer_LSTM, model_bert, model_LSTM



def decode(input_path):

    start_time = time.time()

    SecondVocab = Vocab.SecondVocab()
    ThirdVocab = Vocab.ThirdVocab()
    text = get_text(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), input_path))
    # text = get_text(input_path)

    batch_text = get_batch(text=text,batch_size=20)



    model = model_bert if len(text) <=10000 else model_LSTM
    tokenizer = tokenizer_bert if len(text) <= 10000 else model_LSTM

    triple_info = tripleModel(batch_text, model, tokenizer)  #List<Dict>    # [{text: str}
                                                    # {aspect :[]},
                                                    # {opinion:[]},
                                                    # {triples:[(),()]}
                                                    # {sen_polarity: str}]
                                                    # {target:[(),()]}

    s_time = time.time()
    text_all,triple_all,sen_polarity_all,target_all  = get_all_info(triple_info)

    # aspect_and_opinion = get_a_and_o(aspect_all,opinion_all)
    ao_pair,ao_tri = get_RE_tri(triple_all)

    chart1,chart2,chart3 = get_target_info(target_all,SecondVocab,ThirdVocab)

    # express_info = get_express(express_all,opinion_all)

    e_time = time.time()
    print("Process Post time: %.3f" % (e_time-s_time))
    print('total cost time: %.3f' % (e_time-start_time))
    result = json.dumps( { 'text1': text_all,
                 'text2': triple_info, #每句话中的所有信息
                 'text3': ao_tri, #三元组信息
                 'chart1': chart1,
                 'chart2': chart2,
                 'chart3': chart3,
                })
    return result


def crawler_pro(v1, v2, id):
    try:
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        # options.add_argument("--disbale-gpu")
        # options.add_argument("--proxy-server=http://58.252.200.109:9999")
        web = webdriver.Chrome(options=options)
        web.get("https://hotels.ctrip.com/")

        web.find_element_by_xpath('//*[@id="hotels-destination"]').send_keys(Keys.CONTROL, 'a')
        web.find_element_by_xpath('//*[@id="hotels-destination"]').send_keys(v1)
        WebDriverWait(web, 1, 0.1).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="ibu_hotel_container"]/div[1]/div[1]/div[3]/div/div/ul/li[1]/div/div[3]/div[1]/div/div/p/strong')))
        web.find_element_by_xpath('//*[@id="ibu_hotel_container"]/div[1]/div[1]/div[3]/div/div/ul/li[1]/div/div[3]/div[1]/div/div/p/strong').click()
        print('sleep1')

        web.find_element_by_xpath('//*[@id="ibu_hotel_container"]/div[1]/div[1]/div[3]/div/div/ul/li[3]/div').click()
        web.find_element_by_xpath('//*[@id="keyword"]').click()
        web.find_element_by_xpath('//*[@id="keyword"]').send_keys(Keys.CONTROL, 'a')
        web.find_element_by_xpath('//*[@id="keyword"]').send_keys(v2)
        web.find_element_by_xpath(
            '//*[@id="ibu_hotel_container"]/div[1]/div[1]/div[3]/div/div/ul/li[4]/div/div/div[2]').click()
        WebDriverWait(web, 1, 0.1).until(EC.presence_of_all_elements_located((By.XPATH,'//*[@id="ibu_hotel_container"]/div[1]/div[1]/div[3]/div/div/ul/li[5]/div/i')))
        web.find_element_by_xpath('//*[@id="ibu_hotel_container"]/div[1]/div[1]/div[3]/div/div/ul/li[5]/div/i').click()
        print('sleep2')
        WebDriverWait(web, 3).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="nloginname"]')))
        web.find_element_by_xpath('//*[@id="nloginname"]').send_keys("18434360997")
        print('sleep3')
        web.find_element_by_xpath('//*[@id="npwd"]').send_keys("a123a123")
        driver = crack_my_slide_verification(web)
        driver, characters_pos = crack_my_verification(driver)
        driver = click_verification(driver, characters_pos)
        hotel_Id = get_my_hotelId(driver)
        text = get_hotel_review(hotel_Id, id)
        return text
    finally:
        print('finish')





if __name__ == '__main__':
    load_model()
    input_path = r'input.txt'
    res = decode(input_path)
    print(res)

