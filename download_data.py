from urllib import request
import argparse
import ssl
import os

# HTTP_PROXY = 'http://dev-proxy.oa.com:8080'
# HTTPS_PROXY = 'https://dev-proxy.oa.com:8080'
# os.environ['http_proxy'] = HTTP_PROXY
# os.environ['HTTP_PROXY'] = HTTP_PROXY
# os.environ['https_proxy'] = HTTPS_PROXY
# os.environ['HTTPS_PROXY'] = HTTPS_PROXY

if not os.path.exists("datasets"):
    os.mkdir('datasets')
if not os.path.exists(os.path.join("datasets", "ppi")):
    os.mkdir('datasets/ppi')
if not os.path.exists(os.path.join("datasets", "reddit")):
    os.mkdir('datasets/reddit')
if not os.path.exists(os.path.join("datasets", "cora")):
    os.mkdir('datasets/cora')

ssl._create_default_https_context = ssl._create_unverified_context


download_cora = [os.path.join('datasets', 'cora.zip'),
                 'https://www.dropbox.com/sh/q1ms4e1qgbml6lj/AADO4HoFj5Y76NNoQNNB45Sga?dl=1']
download_ppi = [os.path.join('datasets', 'ppi.zip'),
                'https://www.dropbox.com/sh/brmvu4dnjced6rb/AAArmBA5O_JMIZShlNftqj5Ca?dl=1']
download_reddit = [os.path.join('datasets', 'reddit.zip'),
                   'https://www.dropbox.com/sh/jbodwifw54za0dm/AAAzFV2pDzbGSduvMXqUhPhZa?dl=1']
download_traffic = [[os.path.join('datasets', 'traffic_LA', 'traffic_data.h5'),
                     'https://www.dropbox.com/s/7fbafmbiyjb96n4/df_highway_2012_4mon_sample.h5?dl=1'],
                    [os.path.join('datasets', 'traffic_SF', 'traffic_data.h5'),
                     'https://www.dropbox.com/s/nf6uj5zbfhepgyh/df_highway_2017_6mon_sf.h5?dl=1']]
### temporary use
download_ppi_Graphsage = [os.path.join('datasets', 'ppi_Graphsage.zip'),
                             "https://www.dropbox.com/sh/bw5t70e85no6cae/AAAYNb15UjOv_sjxgCz040PQa?dl=1"]
download_reddit_Graphsage = [os.path.join('datasets', 'reddit_Graphsage.zip'),
                             "https://www.dropbox.com/sh/fa79kg10dwn3w40/AADStZCMmVWkwf3TaHeqDUg5a?dl=1"]


parser = argparse.ArgumentParser(description='Downloading the necessary data')
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help='Whether to overwrite the stored data files')
parser.add_argument('--dataset', type=str, default='cora', help='the dataset name you want to download')

args = parser.parse_args()
download_jobs = []
if args.dataset == "cora":
    download_jobs.append(download_cora)
elif args.dataset == "ppi":
    download_jobs.append(download_ppi)
elif args.dataset == "reddit":
    download_jobs.append(download_reddit)
elif args.dataset == "traffic":
    download_jobs.extend(download_traffic)
elif args.dataset == "all":
    download_jobs.append(download_cora)
    download_jobs.append(download_ppi)
    download_jobs.append(download_reddit)
    download_jobs.extend(download_traffic)
### temporary use
elif args.dataset == "ppi_Graphsage":
    download_jobs.append(download_ppi_Graphsage)
elif args.dataset == "reddit_Graphsage":
    download_jobs.append(download_reddit_Graphsage)

for target_path, src_path in download_jobs:
    if not os.path.exists(target_path) or args.overwrite:
        print('Downloading from %s to %s...' % (src_path, target_path))
        data_file = request.urlopen(src_path)
        with open(target_path, 'wb') as output:
            output.write(data_file.read())
        print('Done!')
    else:
        print('Found %s' % target_path)

def unzip_dataset(data_name):
    if data_name == "cora":
        subprocess.call(["unzip", "datasets/cora.zip", "-d", "datasets/cora"])
        subprocess.call(["rm", "datasets/cora.zip"])
        print("Downloaded the cora dataset!\n")
    elif data_name == "ppi":
        subprocess.call(["unzip", "datasets/ppi.zip", "-d", "datasets/ppi"])
        subprocess.call(["rm", "datasets/ppi.zip"])
        print("Downloaded the ppi dataset!\n")
    elif data_name == "reddit":
        subprocess.call(["unzip", "datasets/reddit.zip", "-d", "datasets/reddit"])
        subprocess.call(["rm", "datasets/reddit.zip"])
        print("Downloaded the reddit dataset!\n")
    elif data_name == "ppi_Graphsage":
        subprocess.call(["unzip", "datasets/ppi_Graphsage.zip", "-d", "datasets/ppi_Graphsage"])
        subprocess.call(["rm", "datasets/ppi_Graphsage.zip"])
        print("Downloaded the ppi_Graphsage dataset!\n")
    elif data_name == "reddit_Graphsage":
        subprocess.call(["unzip", "datasets/reddit_Graphsage.zip", "-d", "datasets/reddit_Graphsage"])
        subprocess.call(["rm", "datasets/reddit_Graphsage.zip"])
        print("Downloaded the reddit_Graphsage dataset!\n")

import subprocess

if args.dataset == "cora" or args.dataset == "all":
    unzip_dataset("cora")
if args.dataset == "ppi" or args.dataset == "all":
    unzip_dataset("ppi")
if args.dataset == "reddit" or args.dataset == "all":
    unzip_dataset("reddit")
if args.dataset == "ppi_Graphsage":
    unzip_dataset("ppi_Graphsage")
if args.dataset == "reddit_Graphsage":
    unzip_dataset("reddit_Graphsage")

subprocess.call(["rm", "-fr", "datasets/__MACOSX"])
