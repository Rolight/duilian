# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from spider.items import SpiderItem
import re


class IduilianSpider(CrawlSpider):
    name = 'iduilian'
    allowed_domains = ['www.iduilian.cn']
    start_urls = ['https://www.iduilian.cn/']
    rules = (
        # Walk URLs
        Rule(
            LinkExtractor(
                allow=(
                    r'www.iduilian.cn/\w+/$',
                    r'www.iduilian.cn/\w+/[a-zA-Z]+_\d+\.html',
                )
            )
        ),
        # Crawl URLs
        Rule(LinkExtractor(
            allow=(
                r'www.iduilian.cn/\w+/\d+\.html',
            )),
            callback='parse_item'
        ),
    )

    def parse_item(self, response):
        self.logger.info('parse url %s' % response.url)
        content = response.css('.detail').extract_first()
        content = re.sub('<.*?>|\t', '', content)
        duilian = re.findall('(.*?)；(.*?)。', content)
        item = SpiderItem()
        item['duilian'] = duilian
        item['content'] = content
        item['url'] = response.url
        return item
