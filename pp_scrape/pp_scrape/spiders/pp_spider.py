from pathlib import Path

import scrapy

class ProjectileSpider(scrapy.Spider):
    name = "projectiles"
    start_urls = [
            "https://www.projectilepoints.net/Points/Merkle.html",
            "https://www.projectilepoints.net/Points/Osceola.html"
    ]

    def parse(self, response):
        raw_urls = response.css("img::attr(src)").getall()
        
        clean_urls = [response.urljoin(url) for url in raw_urls]

        yield {
            "image_urls": clean_urls
        }