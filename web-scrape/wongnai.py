from seleniumbase import Driver

from selenium.webdriver.common.by import By
import time
driver = Driver(uc=True)
url = "https://www.wongnai.com/attractions?regions=373"
driver.uc_open_with_reconnect(url, 10)
driver.uc_gui_click_captcha()
elem = driver.find_element(By.TAG_NAME, "h1")
driver.quit()
time.sleep(60)