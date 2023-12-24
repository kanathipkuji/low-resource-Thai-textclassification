import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()
driver.get('https://www.google.com')

time.sleep(5) # Let the user actually see something!

search_box = driver.find_element_by_name('q')

search_box.send_keys('ChromeDriver')

search_box.submit()

time.sleep(5) # Let the user actually see something!

driver.quit()


# # Find the input field and enter text
# input_element = driver.find_element_by_id('input_field_id')  # Replace 'input_field_id' with the actual ID or another locator of the input field
# input_element.send_keys('Your text goes here')

# # Find the button and click it
# button_element = driver.find_element_by_id('button_id')  # Replace 'button_id' with the actual ID or another locator of the button
# button_element.click()

# # Wait for the next page to load (optional)
# driver.implicitly_wait(10)  # Wait for 10 seconds for the next page to load

# # Perform actions on the next web page
# # ...

# # Close the browser
# driver.quit()