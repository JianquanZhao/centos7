import requests
import time
from lxml import etree
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
#这里是对目标资源的资源定位路径
all_data=[['Title','JD','post date','project type','client country','lot_info','skills and expertise','link']]
Title_loc=                '//div[@class="col-12 cfe-ui-job-details-content"]//header[@class="up-card-header"]//text()'
post_date_loc=            '//span[@id="popper_1"]//div[@class="popper-inner"]/div[@class="popper-content"]/span/text()'
JD_loc=                   '//div[@class="col-12 cfe-ui-job-details-content"]//section[2]//text()'
lot_info_loc_strong=             '//div[@class="col-12 cfe-ui-job-details-content"]//section[3]/ul/li//strong//text()'
lot_info_loc_small=             '//div[@class="col-12 cfe-ui-job-details-content"]//section[3]/ul/li//small//text()'
project_type_loc=         '//div[@class="col-12 cfe-ui-job-details-content"]//section[4]//span//text()'
skills_and_expertise_loc= '//div[@class="col-12 cfe-ui-job-details-content"]//section[5]//div//text()'
client_country_loc='//li[@data-qa="client-location"]/strong/text()'
#function name:login_simulation;
#function     :模拟用户登录，以便浏览器使用用户缓存来自动识别用户身份;
#paras        :login_url 登录界面的url，浏览器使用缓存时能够保证访问某些页面时不用登录自动识别用户id;
#              usename 用户的账户名;
#              password 用户密码;
#return       ;driver 已经被驱动了的浏览器实例，并且是已经完成了获得cookies;
def login_simulation(login_url,username,password):
    login_url='https://www.upwork.com/ab/account-security/login'
    driver=webdriver.Chrome()
    driver.get(login_url)
    time.sleep(4)
    driver.find_element_by_id('login_username').send_keys('3389634510@qq.com')
    driver.find_element_by_id('login_password_continue').click()
    time.sleep(3)
    driver.find_element_by_id('login_password').send_keys('CH12345upwork')
    driver.find_element_by_id('login_control_continue').click()
    time.sleep(6)
    return driver

#function name:get_page_target
#function     :通过已经实例化，完成登录的chrom浏览器，访问目标页面获得源码，从中拿到自己需要的文本内容;
#paras        :driver 驱动了的chrom;
#              link 目标页面的url;
#return       ;目标内容组成的list;
def get_page_target(driver,link):
    all_data=[]
# link='https://www.upwork.com/jobs/Sourcing-Agent-China-Needed-for-Ecommerce-Seller_~010f4917f85f923fec/'
    try:
        #三步走：
        #定位元素
        driver.get(link)
        mouse_put=driver.find_element_by_xpath('//div[@id="posted-on"]//span[@class="text-muted up-popper-trigger"]')
        #模拟放置
        ActionChains(driver).move_to_element(mouse_put).perform()
        time.sleep(2)
        #获得源码
        target_content=driver.page_source
        print('-----------------------获取源码---------------------')
        tree = etree.HTML(target_content)
        print('-----------------开始解析--------------------')
        title_list=tree.xpath(Title_loc)
        Title=''
        for a_title in title_list:
            Title=Title+str(a_title).strip()+'\n'
        # print('Title:',Title)
        JD_list=tree.xpath(JD_loc)
        JD=''
        for a_jd in JD_list:
            JD=JD+str(a_jd).strip()+'\n'
        # print('JD:',JD)
        post_date=''
        post_date_list=tree.xpath(post_date_loc)
        for a_post_date in post_date_list:
            post_date =post_date+str(a_post_date).strip()+'\n'
        # print('post_date:',post_date)
        project_type=''
        project_type_list=tree.xpath(project_type_loc)
        for a_project_type in project_type_list:
            project_type=project_type+str(a_project_type).strip()+'\n'
        # print('project_type:',project_type)

        client_country=''
        client_country_list=tree.xpath(client_country_loc)
        for a_client_country in client_country_list:
            client_country=client_country+str(a_client_country).strip()+'\n'
        # print('client_country:',client_country)

        skills_and_expertise_list=tree.xpath(skills_and_expertise_loc)
        skills_and_expertise=''
        for a_skill in skills_and_expertise_list:
            skills_and_expertise= skills_and_expertise+str(a_skill).strip()+'\n'
        # print('skills_and_expertise:',skills_and_expertise)

        lot_info_loc_small_list = tree.xpath(lot_info_loc_small)
        lot_info_loc_strong_list = tree.xpath(lot_info_loc_strong)
        lot_info = ''

        for i in range(len(lot_info_loc_strong_list)):
            lot_info = lot_info + lot_info_loc_small_list[i] + ':' + lot_info_loc_strong_list[i] + '&'
        # print(lot_info)
        all_data=[Title, JD, post_date, project_type, client_country, lot_info,skills_and_expertise, link]
	signal=1
    except:
        all_data=['', '', '', '', '', '','', link]
	signal=-1
        with open('D:/errot_urls.txt', 'a+') as f:
             f.write(link+'\r')
    else:
    return signal,all_data

def get_all_page(urls_path):
    all_data=[]
    wtih open(urls_path,'r') as f:
        urls=f.readlines()
    for(url in urls)
        if(a_url=='\n'):
            pass
        else:
            signal,a_data=get_page_target(urls)
	    if(signal==1)：
                all_data.append(a_data)    
    all_data_df = pd.DataFrame(all_data)
    all_data_df.to_excel('D:/details.xlsx')
