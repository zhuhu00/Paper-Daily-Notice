# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

from bs4 import BeautifulSoup as bs
import urllib.request

from github_issue import make_github_issue
from config import NEW_SUB_URL, KEYWORD_LIST, USERNAME

def main(args):

    
    # TOKEN = args.token
    
    page = urllib.request.urlopen(NEW_SUB_URL)
    soup = bs(page, 'html.parser')
    content = soup.body.find("div", {'id': 'content'})

    issue_title = content.find("h3").text
    dt_list = content.dl.find_all("dt")
    dd_list = content.dl.find_all("dd")
    arxiv_base = "https://arxiv.org/abs/"

    assert len(dt_list) == len(dd_list)

    keyword_list = KEYWORD_LIST
    keyword_dict = {key: [] for key in keyword_list}

    for i in range(len(dt_list)):
        paper = {}
        paper_number = dt_list[i].text.strip().split(" ")[2].split(":")[-1]
        paper['main_page'] = arxiv_base + paper_number
        paper['pdf'] = arxiv_base.replace('abs', 'pdf') + paper_number

        paper['title'] = dd_list[i].find("div", {"class": "list-title mathjax"}).text.replace("Title: ", "").strip()
        paper['authors'] = dd_list[i].find("div", {"class": "list-authors"}).text.replace("Authors:\n", "").replace(
            "\n", "").strip()
        paper['subjects'] = dd_list[i].find("div", {"class": "list-subjects"}).text.replace("Subjects: ", "").strip()
        paper['abstract'] = dd_list[i].find("p", {"class": "mathjax"}).text.replace("\n", " ").strip()

        for keyword in keyword_list:
            if keyword.lower() in paper['abstract'].lower():
                keyword_dict[keyword].append(paper)

    full_report = '<details>\n    <summary>'+issue_title+'</summary>\n'
    for keyword in keyword_list:
        full_report = full_report + '## Keyword: ' + keyword + '\n'

        if len(keyword_dict[keyword]) == 0:
            full_report = full_report + 'There is no result \n'

        for paper in keyword_dict[keyword]:
            report = '### {}\n - **Authors:** {}\n - **Subjects:** {}\n - **Arxiv link:** {}\n - **Pdf link:** {}\n - **Abstract**\n {}' \
                .format(paper['title'], paper['authors'], paper['subjects'], paper['main_page'], paper['pdf'],
                        paper['abstract'])
            full_report = full_report + report + '\n'

    full_report = full_report + '\n</details>'

    # create an md file using full_report, with the name of date, and upload it to github
    # create a date string
    import datetime
    filename = './Arxiv_Daily_Notice/'+datetime.datetime.now().strftime("%Y-%m-%d") + '-Arxiv-Daily-Paper.md'
    filename_readme = './Arxiv_Daily_Notice/README.md'
    print(filename)
    with open(filename, 'w+') as f:
        f.write(full_report)

    with open(filename_readme, 'w+') as f:
        f.write(full_report)

    make_github_issue(title=issue_title, body=full_report,labels=keyword_list, 
    TOKEN=os.environ['TOKEN'])
    print("end")

if __name__ == '__main__':
    #     处理参数，实际只需要TOKEN
    parser = argparse.ArgumentParser(description='Get the TOKEN for github issue')
    parser.add_argument('-t','--token', help='The github TOKEN', required=True, default='Token Needed!')
    args = vars(parser.parse_args())
    # print(args['token'])
    # print(os.environ['TOKEN'])
    main(args)
