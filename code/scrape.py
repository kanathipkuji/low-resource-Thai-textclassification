import requests
import time
import random
import pandas as pd
import os
from bs4 import BeautifulSoup

cookies = {
    '_ga': 'GA1.1.2115816917.1687512997',
    'search_deka_start_year_cookie': '2560-2562',
    'ci_cookie': 'a%3A4%3A%7Bs%3A10%3A%22session_id%22%3Bs%3A32%3A%225be6a36e4e349323de636d66e40dc251%22%3Bs%3A10%3A%22ip_address%22%3Bs%3A13%3A%22126.227.26.93%22%3Bs%3A10%3A%22user_agent%22%3Bs%3A117%3A%22Mozilla%2F5.0+%28Macintosh%3B+Intel+Mac+OS+X+10_15_7%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F114.0.0.0+Safari%2F537.36%22%3Bs%3A13%3A%22last_activity%22%3Bi%3A1687880515%3B%7Ddcc9f5439d05006f48514f84e82b6db9ea2aef11',
    '_ga_DSQQP3N66C': 'GS1.1.1687879403.5.1.1687879809.0.0.0',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9,th;q=0.8,ja;q=0.7',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    # 'Cookie': '_ga=GA1.1.2115816917.1687512997; search_deka_start_year_cookie=2560-2562; ci_cookie=a%3A4%3A%7Bs%3A10%3A%22session_id%22%3Bs%3A32%3A%225be6a36e4e349323de636d66e40dc251%22%3Bs%3A10%3A%22ip_address%22%3Bs%3A13%3A%22126.227.26.93%22%3Bs%3A10%3A%22user_agent%22%3Bs%3A117%3A%22Mozilla%2F5.0+%28Macintosh%3B+Intel+Mac+OS+X+10_15_7%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F114.0.0.0+Safari%2F537.36%22%3Bs%3A13%3A%22last_activity%22%3Bi%3A1687880515%3B%7Ddcc9f5439d05006f48514f84e82b6db9ea2aef11; _ga_DSQQP3N66C=GS1.1.1687879403.5.1.1687879809.0.0.0',
    'Origin': 'http://deka.supremecourt.or.th',
    'Referer': 'http://deka.supremecourt.or.th/',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'cp-extension-installed': 'Yes',
}

data = {
    'show_item_remark': '1',
    'show_item_primartcourt_deka_no': '1',
    'show_item_deka_black_no': '0',
    'show_item_department': '1',
    'show_item_primarycourt': '1',
    'show_item_judge': '1',
    'show_item_source': '1',
    'show_item_long_text': '1',
    'show_item_short_text': '0',
    'show_item_law': '1',
    'show_item_litigant': '1',
    'show_result_state': '0',
    'search_form_type': 'adv',
    'start': 'true',
    'adv_search_doctype': '',
    'adv_search_word_stext_and_ltext': '',
    'adv_search_deka_start_year': '',
    'adv_search_deka_end_year': '',
    'adv_search_litigant': '',
    'adv_search_judge': '',
    'adv_search_in_judge': '',
    'law_condition': 'AND',
    'adv_search_law_name': 'ประมวลกฎหมายอาญา',
    'adv_search_law_section': '',
    'adv_search_law_paragraph': '',
    'adv_search_law_subsection': '',
    'adv_search_law_other': '',
    'adv_search_black_no_of_scourt': '',
    'adv_search_deka_key': '',
    'adv_search_deka_no': '',
    'adv_search_department': '',
    'adv_search_remark': '',
    'adv_search_law_condition[]': 'AND',
    'adv_search_law[]': 'ประมวลกฎหมายอาญา',
    'adv_search_law_section[]': '',
    'adv_search_law_paragraph[]': '',
    'adv_search_law_subsection[]': '',
    'adv_search_law_other[]': '',
}

output_path = './out/deka_criminal_cases.csv'
num_cases = 0

try:
    df = pd.read_csv(output_path)
    num_cases = len(df)
    print(num_cases)
except:
    pass
# pages = 1319
start_page = 5
end_page = 500

for page in range(start_page, end_page + 1):
    url = f'http://deka.supremecourt.or.th/search/index/{page}'
    response = requests.post(url, cookies=cookies, headers=headers, data=data, verify=False)
    print(f'scraping page: {page}')

    res_list_of_dict = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        lis = soup.find_all(class_='clear result')
        for i, li in enumerate(lis):
            num_cases += 1
            res_dict = {}
            print(f'\tcase {i}')
            c = list(li.children)[0]
            items = c.find_all(class_='item')
            for item in items:
                classes = item.get_attribute_list('class')
                if 'print_item_deka_no' in classes:
                    res_dict['deka_no'] = item.get_text()
                    # deka_nos.append(item.get_text())
                elif 'print_item_litigant' in classes:
                    res_dict['litigant'] = item.get_text()
                    # litigants.append(item.get_text())
                elif 'print_item_law' in classes:
                    res_dict['law'] = item.get_text()
                    # laws.append(item.get_text())
                elif 'print_item_short_text' in classes:
                    res_dict['short_text'] = item.get_text()
                    # short_texts.append(item.get_text())
                elif 'print_item_long_text' in classes:
                    res_dict['long_text'] = item.get_text()
                    # long_texts.append(item.get_text())
                elif 'print_item_judge' in classes:
                    res_dict['judge'] = item.get_text()
                    # judges.append(item.get_text())
                elif 'print_item_primarycourt' in classes:
                    res_dict['primary_court'] = item.get_text()
                    # primary_courts.append(item.get_text())
                elif 'print_item_source' in classes:
                    res_dict['source'] = item.get_text()
                    # sources.append(item.get_text())
                elif 'print_item_department' in classes:
                    res_dict['department'] = item.get_text()
                    # departments.append(item.get_text())
                elif 'print_item_deka_black_no' in classes:
                    res_dict['deka_black_no'] = item.get_text()
                    # black_nos.append(item.get_text())
                elif 'print_item_primartcourt_deka_no' in classes:
                    text = item.get_text()
                    if text.startswith('หมายเลขคดีดำ'):
                        res_dict['primary_black_no'] = item.get_text()
                        # primary_black_nos.append(item.get_text())
                    else:
                        res_dict['primary_red_no'] = item.get_text()
                        # primary_red_nos.append(item.get_text())
                elif 'print_item_remark' in classes:
                    res_dict['remarks'] = item.get_text()
                    # remarks.append(item.get_text())
            res_list_of_dict.append(res_dict)
        time.sleep(random.randint(3, 7))
    else:
        print('Failed to retrieve the webpage')

    df = pd.DataFrame(res_list_of_dict, index=pd.RangeIndex(start=num_cases-len(lis), stop=num_cases))
    df.to_csv(output_path, mode='a', encoding='utf-8', index=True, header=not os.path.exists(output_path))
    # print(res_list_of_dict)
