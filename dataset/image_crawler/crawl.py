import logging
import os.path as osp
from argparse import ArgumentParser

from icrawler.builtin import (BaiduImageCrawler, BingImageCrawler,
                              FlickrImageCrawler, GoogleImageCrawler,
                              GreedyImageCrawler, UrlListCrawler)


_CRAWLER_NAME_GOOGLE = 'google'
_CRAWLER_NAME_BING = 'bing'
_CRAWLER_NAME_BAIDU = 'baidu'

def build_crawler(crawler, sub_dir=""):
    new_crawler = None
    if crawler == _CRAWLER_NAME_GOOGLE:
        new_crawler = GoogleImageCrawler(
            downloader_threads=4,
            storage={'root_dir': 'images/'+ sub_dir +'/google'},
            log_level=logging.INFO)
    elif crawler == _CRAWLER_NAME_BING:
        new_crawler = BingImageCrawler(
            downloader_threads=2,
            storage={'root_dir': 'images/'+ sub_dir +'/bing'},
            log_level=logging.INFO)
    elif crawler == _CRAWLER_NAME_BAIDU:
        new_crawler = BaiduImageCrawler(
            downloader_threads=4, storage={'root_dir': 'images/'+ sub_dir +'/baidu'})
    
    return new_crawler


def build_filter(args):
    return dict(license='commercial,modify')


# def test_flickr():
#     print('start testing FlickrImageCrawler')
#     flickr_crawler = FlickrImageCrawler(
#         apikey=None,
#         parser_threads=2,
#         downloader_threads=4,
#         storage={'root_dir': 'images/flickr'})
#     flickr_crawler.crawl(
#         max_num=10,
#         tags='family,child',
#         tag_mode='all',
#         group_id='68012010@N00')


def main():
    parser = ArgumentParser(description='Test built-in crawlers')
    parser.add_argument('--crawler', nargs='+', default=['google', 'bing', 'baidu'])
    parser.add_argument("--name")
    parser.add_argument("--max", type=int, default=1)
    args = parser.parse_args()
    
    crawlers = [build_crawler(crawler, args.name) for crawler in args.crawler]

    print("name: %s, max:%d" %(args.name, args.max))

    for crawler in crawlers:
        crawler.crawl(args.name, max_num=args.max)


if __name__ == '__main__':
    main()

