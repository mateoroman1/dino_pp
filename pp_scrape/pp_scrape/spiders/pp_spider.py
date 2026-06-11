from pathlib import Path

import scrapy

from dotenv import load_dotenv
import os

load_dotenv()

site_list = os.getenv("PP_PAGES").split(",")

class ProjectileSpider(scrapy.Spider):
    name = "projectiles"
    start_urls = site_list

    def parse(self, response):
        raw_urls = response.css("img::attr(src)").getall()
        
        clean_urls = [response.urljoin(url) for url in raw_urls]

        yield {
            "image_urls": clean_urls
        }