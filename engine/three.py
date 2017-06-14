#!/usr/bin/env python
# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import time
import random
import numpy as np
import matplotlib.pyplot as plt

class PoorMansGymEnv(object):
    class ActionSpace(object):
        def __init__(self):
            self.actions = [Keys.LEFT, Keys.RIGHT, Keys.DOWN]#, Keys.UP]
            self.n = len(self.actions)

        def sample():
            return random.choice(self.actions)
    def __init__(self):
        self.scores = []
        self.i = 0
        self.lastscore = 0
        self.action_space = self.ActionSpace()
        self.driver = webdriver.Chrome()
        self.url = 'https://goo.gl/zS3bt3'
        self.driver.get(self.url)

        self.driver.find_element_by_class_name('child-pushSpace').click()
        self.driver.find_element_by_name('game').send_keys(Keys.LEFT)
        self.driver.switch_to_frame(self.driver.find_element_by_name("game"))

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel('steps')
        self.ax.set_ylabel('score')
        plt.show(block=False)

    def grid(self):
        tmp = np.array(self.driver.execute_script("return game.grid;"))
        self.new = (tmp == self.lastgrid).all()
        # self.lastgrid = tmp
        self.lastgrid = np.log(tmp+1)/9

    def replot(self):
        self.i += 1
        self.line.set_data(range(len(self.scores)), self.scores)
        self.ax.set_xlim(0, len(self.scores))
        self.ax.set_ylim(0, max(self.scores))
        self.fig.canvas.draw()

    def reset(self):
        self.scores.append(self.lastscore)
        self.replot()
        self.driver.execute_script("game.newGame();")
        self.lastgrid = 16*[0]
        self.lastscore = 0

        replace_function = """
        ThreesGame.prototype.plantFood = function(t) {
            var e = void 0
              , o = void 0
              , n = void 0
              , i = void 0
              , r = [];
            t[0] ? (e = t[0] > 0 ? this.gridSize - 1 : 0,
            n = e,
            o = 0,
            i = this.gridSize - 1) : (o = t[1] > 0 ? this.gridSize - 1 : 0,
            i = o,
            e = 0,
            n = this.gridSize - 1);
            for (var s = e; s <= n; s++)
                for (var a = o; a <= i; a++) {
                    var u = [s, a];
                    this.getGrid(u) || r.push(u)
                }
            var c = r[Math.floor(0 * r.length)];
            this.setGrid(c, this.food),
            this.animations.push({
                type: "new",
                num: this.food,
                where: c
            }),
            this.food = 1 == this.food ? 2 : 1;
        }
        """
        self.driver.execute_script(replace_function)

        return np.array(self.lastgrid)

    def done(self):
        return 0 == self.driver.execute_script("return game.canMove.size;")

    def score(self):
        newscore = self.driver.execute_script("return game.score;")
        # diff = newscore - self.lastscore

        diff = 1
        if newscore == self.lastscore:
            diff = 0

        # diff += 1
        self.lastscore = newscore
        return diff

    def seed(self, xy):
        pass

    def step(self, key):
        self.driver.find_element_by_xpath('//body').send_keys(self.action_space.actions[key])
        self.grid()
        done = self.done()
        reward = self.score()
        # time.sleep(0.2)
        return (self.lastgrid, reward, done, {})

    def render(self):
        pass

    def __del__(self):
        self.driver.close()

# time.sleep(1)

# keys = [Keys.LEFT, Keys.RIGHT, Keys.DOWN, Keys.UP]
# keys = []
# keys += 1*[Keys.LEFT]
# keys += 1*[Keys.RIGHT]
# keys += 1*[Keys.DOWN]
# keys += 1*[Keys.UP]

# scores = []
# while True:
    # sendkey(random.choice(keys))

    # # printgrid()

    # if isdead():
        # scores.append(score())
        # print(max(scores))
        # restart()

# time.sleep(3)
