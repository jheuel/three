#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
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
import os

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

class PoorMansGymEnv(object):
    class ActionSpace(object):
        def __init__(self):
            self.actions = [Keys.DOWN, Keys.UP, Keys.LEFT]
            self.blocking_moves = [Keys.DOWN, Keys.UP]
            self.n = len(self.actions)

    def __init__(self, average_over=40):
        self.im_size = (64, 64)
        self.action_space = self.ActionSpace()
        self.url = 'https://www.gameeapp.com/game/0fmMKCljn~telegram:inline~759394035287578674~3761925~Johannes~AgAAAD5FAAAbtCgRddGKewLJ2RQ#tgShareScoreUrl=tg%3A%2F%2Fshare_game_score%3Fhash%3DgCUIA3WqOjLIrVP2y1tHh1eLxXuG3Rbl470r7JzXQn8'
        self.driver = None

        self.last_score = 0
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

    def reset(self):
        print(' score: {}'.format(self.last_score))
        if not self.driver is None and len(self.scores) % 100 == 0:
            self.driver.quit()
            os.system('pkill firefox')
            self.driver = None

        if self.driver is None:
            # self.driver = webdriver.Chrome()
            self.driver = webdriver.Firefox()
            self.driver.get(self.url)
            self.driver.set_window_position(0, 0)
            self.driver.set_window_size(700, 800)
            time.sleep(3.0)
            self.click_overlay()

        # self.driver.switch_to_default_content()
        # self.driver.switch_to_frame(self.driver.find_element_by_name("game"))
        self.scores.append(self.score())
        self.replot()
        time.sleep(3)

        self.driver.switch_to_default_content()
        actions = ActionChains(self.driver)
        actions.key_down('r')
        actions.perform()
        time.sleep(0.5)
        actions.key_up('r')
        actions.perform()

        self.click_overlay()

        self.last_score = 0
        self.last_health = self.health()

        return self.obs()

    def replot(self):
        self.line.set_data(range(len(self.scores)), self.scores)

        if len(self.scores) % self.average_over == 0:
            averages = np.mean(
                np.array(self.average_over * [0] + self.scores).reshape(
                    -1, self.average_over),
                axis=1)
            self.avgline.set_data(
                self.average_over * np.array(range(len(averages))), averages)
            self.fig.savefig('/home/jheuel/Dropbox/dogfight.pdf')

        self.ax.set_xlim(0, len(self.scores))
        self.ax.set_ylim(0, max(self.scores))
        self.fig.canvas.draw()


    def press_space(self):
        # self.driver.switch_to_frame(self.driver.find_element_by_name("game"))
        self.driver.switch_to_default_content()
        actions = ActionChains(self.driver)
        actions.key_down(Keys.SPACE)
        actions.perform()
        time.sleep(0.5)
        actions.key_up(Keys.SPACE)
        actions.perform()

    def click_overlay(self):
        self.press_space()

        class_names = [
            # 'this-inner',
            'this-bg',
            # 'replay',
            # # 'gameReady',
            # 'startOverlay',
        ]

        for class_name in class_names:
            self.driver.execute_script("return document.getElementsByClassName('{}')[0].style.display = 'none';".format(class_name))

    def sample(self):
        return random.randint(0, self.action_space.n-1)

    def keypress(self, k):
        # https://stackoverflow.com/a/44543367 could help with chrome

        actions = ActionChains(self.driver)
        actions.key_down(k)
        actions.perform()
        time.sleep(0.1)
        actions.key_up(k)
        actions.perform()

        # if k in self.action_space.blocking_moves:
            # time.sleep(0.5)

    def score(self):
        s = int(self.driver.find_element_by_class_name('child-score').text)
        return s

    def health(self):
        self.driver.switch_to_frame(self.driver.find_element_by_name("game"))
        health = self.driver.execute_script('return game.state.callbackContext.player.health')
        self.driver.switch_to_default_content()

        return health

    def done(self):
        return 0 == self.health()

    def seed(self, xy):
        pass

    def obs(self):
        self.driver.switch_to_frame(self.driver.find_element_by_name("game"))
        b = BytesIO(base64.b64decode(self.driver.get_screenshot_as_base64()))
        im = Image.open(b)

        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        im = im.crop(bbox)
        im.thumbnail(self.im_size, Image.ANTIALIAS)
        # im.save('test.png')
        # im.show()
        self.driver.switch_to_default_content()
        return im

    def step(self, key):
        self.keypress(self.action_space.actions[key])

        self.driver.find_element_by_xpath('//body').send_keys(
            self.action_space.actions[key])

        im = self.obs()

        new_score = self.score()
        new_health = self.health()
        reward = new_score - self.last_score# - (self.last_health - new_health)
        # reward = 0.025 + new_score - self.last_score - (self.last_health - new_health)
        self.last_score = new_score
        self.last_health = new_health

        done = self.done()
        observation = im
        return (observation, reward, done, {'score': self.score()})

    def __del__(self):
        self.driver.quit()
