# Get-daily-arxiv-notice

## TODO

- [x] 不显示github的token
  在secret中添加token的编号，便能够在每次github action中传入token了。





You can get daily arxiv notification with pre-defined keywords as [here](https://github.com/kobiso/daily-arxiv-noti/issues).

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

#### 1. Create a Repo

Create a repository to get notification in your github.

#### 2. Set Config

Revise `config.py` as your perferences.

```python
# Authentication for user filing issue (must have read/write access to repository to add issue to)
USERNAME = 'changeme'
TOKEN = 'changeme'

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
