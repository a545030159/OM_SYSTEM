from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
import time
import json
import requests
from selenium.webdriver.common.action_chains import ActionChains
import re
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# opt = Options()
# opt.add_argument("--headless")
# opt.add_argument("--disbale-gpu")
import threading




def crack_my_slide_verification(driver):
    WebDriverWait(driver, 1, 0.1).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sliderddnormal"]/div[1]/div[2]')))
    slider_btn = driver.find_element_by_xpath('//*[@id="sliderddnormal"]/div[1]/div[2]')
    actions = ActionChains(driver)
    actions.click_and_hold(slider_btn).perform()
    actions.move_by_offset(280,0).release(slider_btn).perform()
    print('sleep4')
    return driver


def crack_my_verification(driver):
    WebDriverWait(driver, 0.5, 0.1).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sliderddnormal-choose"]/div[2]/div[4]/div/a')))
    driver.find_element_by_xpath('//*[@id="sliderddnormal-choose"]/div[2]/div[4]/div/a').click()
    print('sleep5')
    dest_img_url = driver.find_element_by_xpath('//*[@id="sliderddnormal-choose"]/div[2]/div[1]/img').get_attribute('src')
    dest_img_url = dest_img_url.replace('data:image/jpg;base64,', '')
    sele_img_url = driver.find_element_by_xpath('//*[@id="sliderddnormal-choose"]/div[2]/div[3]/img').get_attribute('src')
    sele_img_url = sele_img_url.replace('data:image/jpg;base64,', '')
    url = "http://api.ttshitu.com/predict"
    data = {
            "username": 'a545030159',
            "password": 'a123a123',
            "typeid": '54',
            "title_image": dest_img_url,
            "image": sele_img_url,
            }
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
    }
    r = requests.post(url=url, data=json.dumps(data), headers=headers).json()
    characters_pos = r['data']['result']
    characters_pos = characters_pos.split('|')
    return driver, characters_pos



def click_verification(driver,characters_pos):
    crack_validation(driver, characters_pos)
    check_result_span = driver.find_element_by_xpath('//*[@id="sliderddnormal"]/div[1]/div[4]/span')
    while(u'校验成功' not in check_result_span.text):
        driver.find_element_by_xpath('//*[@id="sliderddnormal-choose"]/div[2]/div[4]/div/a').click()
        driver, characters_pos = crack_my_verification(driver)
        crack_validation(driver, characters_pos)
    return driver


def crack_validation(driver,characters_pos):
    actions = ActionChains(driver)
    cpt_big_img = driver.find_element_by_class_name("cpt-big-img")
    characters_pos_processed = [tuple(pos.split(','))for pos in characters_pos]
    actions.move_to_element_with_offset(cpt_big_img, 0, 0).perform()
    for i in range(0, len(characters_pos_processed)):
        x, y = int(characters_pos_processed[i][0]), int(characters_pos_processed[i][1])
        for j in range(i):
            actions.move_to_element_with_offset(cpt_big_img,x, y).perform()
        actions.move_to_element_with_offset(cpt_big_img,x, y).click().perform()
    driver.find_element_by_xpath('//*[@id="sliderddnormal-choose"]/div[2]/div[4]/a').click()


def get_my_hotelId(browser):
    driver = browser
    WebDriverWait(driver, 1, 0.1).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="nsubmit"]')))
    driver.find_element_by_xpath('//*[@id="nsubmit"]').click()
    WebDriverWait(driver, 3, 0.1).until(EC.presence_of_all_elements_located((By.XPATH, '/html/body/div[2]/div/section/div[2]/div/div/div[1]/div/div[1]/div[1]/div[2]/div[5]/div[10]/div')))
    text = driver.find_element_by_xpath('/html/body/div[2]/div/section/div[2]/div/div/div[1]/div/div[1]/div[1]/div[2]/div[5]/div[10]/div').get_attribute('id')
    print('sleep6')
    s = re.search(r"\d+", text)
    hotel_Id = s.group()
    return hotel_Id


def get_hotel_review(hotel_id,id):
    num = 0
    return_text = ''
    hotelID = hotel_id
    userip = "120.236.174.150"
    if int(id) == 0:
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input.txt')
    else:
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input_1.txt')
    error_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'error.txt')
    log = open(file_path, 'w', encoding='utf-8')
    errorLog = open(error_path, 'a')
    requestURL = "https://m.ctrip.com/restapi/soa2/16709/json/GetReviewList"
    pageNo = 0
    print("开始爬取hotelID=" + str(hotelID) + "酒店的评论")
    while (True):
        # RequestPayload是通过抓包得到的Ajax请求的提交表单
        # 通过json.dumps，其中的True、False、None都会转成标准的true、false、null，不必担心
        pageNo += 1
        requestPayload = {"PageNo": pageNo,
                          "PageSize": 10,
                          "MasterHotelId": hotelID,  # hotel的ID，可以体现在主页URL上
                          "NeedFilter": True,
                          "UnUsefulPageNo": 1,
                          "UnUsefulPageSize": 5,
                          "isHasFold": False,
                          "head":
                              {"Locale": "zh-CN",
                               "Currency": "CNY",
                               "Device": "PC",
                               "UserIP": userip,  # 发起请求的IP地址
                               "Group": "",
                               "ReferenceID": "",
                               "UserRegion": "CN",
                               "AID": "4897",  #
                               "SID": "155952",  #
                               "Ticket": "",
                               "UID": "",
                               "IsQuickBooking": "",
                               "ClientID": "1619451283416.eo1bls",  #
                               "OUID": "index",
                               "TimeZone": "8",
                               "P": "49241000217",  #
                               "PageID": "102003",  #
                               "Version": "",
                               "HotelExtension":
                                   {"WebpSupport": True,
                                    "group": "CTRIP",
                                    "Qid": None,
                                    "hasAidInUrl": False},
                               "Frontend": {"vid": "1619451283416.eo1bls", "sessionID": 4, "pvid": 13}},  #
                          "ServerData": ""
                          }
        data = json.dumps(requestPayload)
        # 当爬取失败的时候，休眠1s后重新爬取即可
        try:
            html = requests.post(requestURL, data=data)
            html.encoding='utf-8'
        except Exception as e:
            print('爬取第' + str(pageNo) + '页评论失败，异常：' + str(e))
            errorLog.write('爬取第' + str(pageNo) + '页评论失败，异常：' + str(e))
            time.sleep(1)
            pageNo -= 1
            continue
        # print(RequestPayload)
        # print(pageNo)
        if (html.status_code != 200):
            print(('爬取第' + str(pageNo) + '页评论失败，status_code =' + str(html.status_code)))
        else:
            if 'ReviewList' not in json.loads(html.text)['Response'].keys():
                print('已完成爬取hotelID=' + str(hotelID) + "酒店的评论")
                break
            print('爬取第' + str(pageNo) + '页评论成功')
            reviewList = json.loads(html.text)['Response']['ReviewList']

            for singleReview in reviewList:
                if not singleReview['reviewDetails']['reviewContent'].strip():
                    continue
                reviewString = singleReview['reviewDetails']['reviewContent'].strip().replace("\n", "")
                num += 1
                log.write(reviewString+"\n")
                return_text += (str(num)+'.' + reviewString + '\n')
    log.close()
    errorLog.close()
    return return_text