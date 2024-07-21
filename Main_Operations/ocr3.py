import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

'''
Download the appropriate chromeDriver from https://googlechromelabs.github.io/chrome-for-testing/#stable
Python 3.9.6
pip install selenium
'''

def ocr_func(path: str,):
    chrome_driver_path = r'C:\Users\divya\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe'
    img_file_path = path
    url = 'https://ocr.sanskritdictionary.com/'  # Replace with your URL

    # if not os.path.exists(chrome_driver_path):
    #     raise FileNotFoundError(f"ChromeDriver not found at {chrome_driver_path}")

    # Initialize WebDriver
    service = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    # Use below options to run in headless mode. This was not working for me though, needs some research
    # options.add_argument('--headless')
    # options.add_argument('--disable-gpu')  # Required for headless mode to work in some environments
    # options.add_argument('--no-sandbox')
    # options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Open the webpage
        driver.get(url)

        print("Trying to upload a file")
        # Wait for the page to load and the link to be clickable
        upload_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'upload')]"))
        )
        print("Clicking the upload link")
        # Click the upload link to open the file dialog
        upload_link.click()

        # Wait for the file input to be visible and interactable
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "pictureFile"))
        )
        print("Image file uploaded")
        # Upload the file
        file_input.send_keys(img_file_path)
        print("Waiting for response")
        # Allow some time for the upload to complete and response to be processed
        time.sleep(15)

        # Get the page source after file upload
        iframe = driver.find_element(By.ID, "tinymcetext_ifr")
        driver.switch_to.frame(iframe)

        # Now you can interact with elements inside the iframe
        # For example, to get the content of a text area inside the iframe
        text_area_content = driver.find_element(By.TAG_NAME, "body").text
        print(text_area_content)


    finally:
        # Close the browser
        driver.quit()

    return text_area_content