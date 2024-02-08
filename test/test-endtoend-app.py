import unittest
from flask import request
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import re


class TestAppE2E(unittest.TestCase):
    def setUp(self):
        # Launch your flask app first

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.binary_location = "/usr/local/bin/chromedriver"
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get('http://localhost:5000')

    def test_add_and_delete_item(self):
        # you can use the driver to find elements in the page
        # example:
        playlist_link = r"https://open.spotify.com/playlist/37i9dQZF1DWXdiK4WAVRUW"
        input_field = self.driver.find_element(
            By.XPATH, "//input[@name='playlist']")
        # this refers to the input with 'name="playlist"' attribute
        # checkout the rest of the methods in the documentation:
        # https://selenium-python.readthedocs.io/locating-elements.html

        # after you select your element, you can send it a key press:

        input_field.send_keys(playlist_link)
        input_field.send_keys(Keys.RETURN)

        # and you can use the rest of the assetion methods as well:

        predicted_songs_tags = re.findall(
            r"<iframe(.*?)<\/iframe>", self.driver.page_source)

        print(predicted_songs_tags)
        self.assertEqual(len(predicted_songs_tags), 10)

    def tearDown(self):
        self.driver.close()


if __name__ == '__main__':
    unittest.main()
