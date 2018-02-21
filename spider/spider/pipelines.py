# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
import pymongo


class SpiderPipeline(object):
    def open_spider(self, spider):
        self.file = open('items.json', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item), ensure_ascii=False) + "$$#$\n"
        self.file.write(line)
        return item


"""
class SpiderPipeline(object):

    collection_name = 'iduilian'

    def open_spider(self, spider):
        host = spider.settings['MONGODB_HOST']
        port = spider.settings['MONGODB_PORT']
        dbName = spider.settings['MONGODB_DBNAME']
        self.client = pymongo.MongoClient(host=host, port=port)
        self.db = self.client[dbName]
        self.post = self.db[spider.settings['MONGODB_DOCNAME']]

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
        self.post.insert_one(dict(item))
        return item
"""
