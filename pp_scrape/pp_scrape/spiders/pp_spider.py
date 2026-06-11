from pathlib import Path

import scrapy

class ProjectileSpider(scrapy.Spider):
    name = "projectiles"
    start_urls = [
            "https://www.projectilepoints.net/Search/Kansas_Notched.html"
    ]

    def parse(self, response):
        raw_urls = response.css("img::attr(src)").getall()
        
        clean_urls = [response.urljoin(url) for url in raw_urls]

        yield {
            "image_urls": clean_urls
        }