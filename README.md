# Get-daily-arxiv-notice

> 不知道什么原因，不能更新issue了，本地运行返回401，action也不能正常更新issue，不过还是会更新到markdown里面的，每天论文都是在文件夹`Arxiv_Daily_Notice`文件夹下`xxxx-xx-xx-Arxiv-Daily-Paper.md`，文件夹的`README.md`是最新更新的论文
> 
> 以及，如果感觉以后论文太多翻的麻烦，可以在个人主页看：[**https://zhuhu00.github.io/blog/**](https://zhuhu00.github.io/blog/)，这里会更新每天的arxiv有关SLAM等的文章

# 如何使用
1. `fork`本repository，然后在Setting->Security->Secrets->Actions下，创建一个`Repository secrets`, 并记下名字为`ISSUE_TOKEN`,这个TOKEN上需要先做github账号下申请的。然后粘贴到`ISSUE_TOKEN`。
2. 修改`config.py`下，repo的名字，以及github名字等，可查看后面的内容。
3. 可先在本地运行，成功后github的action会每天自动运行

You can get daily arxiv notification with pre-defined keywords as [here](https://github.com/zhuhu00/arxiv-daily-notice/issues).

Arxiv.org announces new submissions every day on fixed time as informed [here](https://arxiv.org/help/submit).

This repository makes it easy to filter papers and follow-up new papers which are in your interests by creating an issue in a github repository.


## Prerequisites

- Python3.x

Install requirements with below command.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Usage

### 1. Create a Repo

Create a repository to get notification in your github.

### 2. Set Config

Revise `config.py` as your perferences.

```python
# Authentication for user filing issue (must have read/write access to repository to add issue to)
USERNAME = 'changeme'

# The repository to add this issue to
REPO_OWNER = 'changeme'
REPO_NAME = 'changeme'

# Set new submission url of subject
NEW_SUB_URL = 'https://arxiv.org/list/cs/new'

# Keywords to search
KEYWORD_LIST = ["changeme"]
```

#### 3. Set Cronjob

You need to set a cronjob to run the code everyday to get the daily notification.

Refer the [announcement schedule](https://arxiv.org/help/submit) in arxiv.org and set the cronjob as below.

```bash
$ cronjob -e
$ 0 13 * * mon-fri python PATH-TO-CODE/get-daily-arxiv-noti/main.py
```

## 设定定时任务时可以怎么设置呢？

ubuntu下可以是crontab，也是一样的功能

# Arxiv上的时区

![](https://gitee.com/zhuhu00/img/raw/master/2021-10/20210606092857.png)
