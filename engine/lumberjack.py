#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import random
import numpy as np
from PIL import Image, ImageChops
from io import BytesIO
import base64

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

class PoorMansGymEnv(object):
    class ActionSpace(object):
        def __init__(self):
            self.actions = [Keys.RIGHT, Keys.LEFT]
            self.n = len(self.actions)

    def __init__(self, average_over=20):
        self.action_space = self.ActionSpace()
        # self.driver = webdriver.Chrome()
        self.driver = webdriver.Firefox()
        # self.url = 'https://tbot.xyz/lumber/#eyJ1IjoyODc4ODAyMTksIm4iOiJuaWtsYXMgbm9sdGUiLCJnIjoiTHVtYmVySmFjayIsImNpIjoiMjI3OTUwMjExODA1NzEwNzg5MiIsImkiOiJBZ0FBQUZoSUFBQWJ0Q2dSalJUX3hIVDFtaDAifTQ0ODFkYmFlNzM0ZTUwNWViZTdhYzY4YzhiY2IzZDBh'
        self.url = 'https://tbot.xyz/lumber/#eyJ1IjozNzYxOTI1LCJuIjoiSm9oYW5uZXMgSGV1ZWwiLCJnIjoiTHVtYmVySmFjayIsImNpIjoiMjI3OTUwMjExODA1NzEwNzg5MiIsImkiOiJBZ0FBQUZoSUFBQWJ0Q2dSalJUX3hIVDFtaDAifTU2ZmM5OWFhZDkzMDZjZTE5ZTRhNmJkNzJmZmUxODcx?tgShareScoreUrl=tg%3A%2F%2Fshare_game_score%3Fhash%3Do-BN6mPgKwSCkj39r4rr4RjOPGmNdMf7TQaG1HQm4ks'
        self.driver.get(self.url)
        self.driver.set_window_position(0, 0)
        self.driver.set_window_size(700, 800)
        time.sleep(1.)
        self.click_overlay()

        self.scores = []
        self.average_over = average_over
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'x', lw=1, label='1 game')
        self.avgline, = self.ax.plot(
            [], [], lw=2, label='{} game average'.format(self.average_over))
        self.ax.set_xlabel('games')
        self.ax.set_ylabel('score')
        self.ax.legend(loc='best', ncol=2)
        self.lasttile = 1
        plt.show(block=False)

    def replot(self):
        self.line.set_data(range(len(self.scores)), self.scores)

        if len(self.scores) % self.average_over == 0:
            averages = np.mean(
                np.array(self.average_over * [0] + self.scores).reshape(
                    -1, self.average_over),
                axis=1)
            self.avgline.set_data(
                self.average_over * np.array(range(len(averages))), averages)
            self.fig.savefig('lumber.pdf')

        self.ax.set_xlim(0, len(self.scores))
        self.ax.set_ylim(0, max(self.scores))
        self.fig.canvas.draw()


    def press_space(self):
        # self.driver.switch_to_frame(self.driver.find_element_by_name("game"))
        self.driver.switch_to_default_content()
        actions = ActionChains(self.driver)
        actions.key_down(Keys.SPACE)
        actions.perform()
        time.sleep(0.2)
        actions.key_up(Keys.SPACE)
        actions.perform()

    def click_overlay(self):
        pass
#        self.press_space()
#
#        class_names = [
#            # 'this-inner',
#            #'this-bg',
#            # 'replay',
#             'gameReady',
#            # 'startOverlay',
#        ]
#
#        for class_name in class_names:
#            self.driver.execute_script("return document.getElementsByClassName('{}')[0].style.display = 'none';".format(class_name))

    def sample(self):
        return random.randint(0, self.action_space.n-1)

    def reset(self):
        self.scores.append(self.score())
        print('reset')
        
        if not self.driver is None and len(self.scores) % 500 == 0:
            self.driver.quit()
            os.system('pkill firefox')
            self.driver = None

        if self.driver is None:
            # self.driver = webdriver.Chrome()
            self.driver = webdriver.Firefox()
            self.driver.get(self.url)
            self.driver.set_window_position(0, 0)
            self.driver.set_window_size(700, 900)
            time.sleep(.5)
            self.click_overlay()
        
        self.replot()
        time.sleep(.5)

        self.driver.switch_to_default_content()
        self.press_space()
        self.click_overlay()

        self.last_score = 0

        return self.obs()

    def keypress(self, k):
        self.driver.find_element_by_class_name('buttons').send_keys(k)

    def score(self):
        return int(self.driver.find_element_by_xpath('//*[@id="score_value"]').get_attribute('innerHTML'))
    
    def done(self):
        return 'in_result' in self.driver.find_element_by_id('page_wrap').get_attribute('class')

    def seed(self, xy):
        pass

    def obs(self):
        #self.driver.switch_to_frame(self.driver.find_element_by_name(""))
        b = BytesIO(base64.b64decode(self.driver.get_screenshot_as_base64()))
        im = Image.open(b)

        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        # print(bbox)
        im = im.crop(bbox)
        im.thumbnail((84,84), Image.ANTIALIAS)
        # im.save('test.png')
        # print(im.size)
        #im.show()
        self.driver.switch_to_default_content()
        return im

    def step(self, key):
        self.keypress(self.action_space.actions[key])
        self.keypress(self.action_space.actions[key])
        # self.driver.save_screenshot('test.png')
        # points = [rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']]

#        self.driver.find_element_by_xpath('//body').send_keys(
#            self.action_space.actions[key])
        # time.sleep(.05)
        im = self.obs()
        self.new_score = self.score()
        reward = -1 if self.done() else 1 #self.new_score - self.last_score
        self.last_score = self.new_score
        done = self.done()
        observation = im
        return (observation, reward, done, {'score': self.score()})

    def __del__(self):
        self.driver.close()
