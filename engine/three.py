#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
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
            self.directions = [Keys.LEFT, Keys.RIGHT, Keys.DOWN, Keys.UP]
            self.actions = [Keys.LEFT, Keys.RIGHT, Keys.DOWN]#, Keys.UP]
            self.n = len(self.actions)

    def __init__(self, stop_on_invalid_move=False, average_over=15, max_invalid_moves=5, square_grid=False):
        self.square_grid = square_grid
        self.stop_on_invalid_move = stop_on_invalid_move
        self.max_invalid_moves = max_invalid_moves
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

        self.average_over = average_over
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2, label='1 game')
        self.avgline, = self.ax.plot(
            [], [], lw=2, label='{} game average'.format(self.average_over))
        self.ax.set_xlabel('games')
        self.ax.set_ylabel('score')
        self.ax.legend(loc='best', ncol=2)
        self.lasttile = 1
        plt.show(block=False)

    def simulate(self, direction):
        key = self.action_space.directions[direction]
        tmpgrid = np.array(self.lastgrid).reshape(-1, 4)
        x = {
                Keys.LEFT:  lambda i,j: ((i,j), (i,j+1), (i,j+2)),
                Keys.RIGHT: lambda i,j: ((i,3-j), (i,3-(j+1)), (i,3-(j+2))),
                Keys.UP:    lambda j,i: ((i,j), (i+1,j), (i+2,j)),
                Keys.DOWN:  lambda j,i: ((3-i,j), (3-(i+1),j), (3-(i+2),j)),
        }


        def canjoin(grid, a, b):
            return tmpgrid[a] < 3 and tmpgrid[b] < 3 and tmpgrid[a] != tmpgrid[b] or tmpgrid[a] > 2 and tmpgrid[a] == self.lastscore[b]

        for i in range(4):
            for j in range(3):
                a, b, c = x[key](i,j)
                if tmpgrid[a] == 0:
                    if max(c) < 4 and min(c) >= 0 and tmpgrid[b] > 0 and canjoin(tmpgrid, b, c):
                        tmpgrid[a] = tmpgrid[b] + tmpgrid[c]
                        tmpgrid[b] = 0
                        tmpgrid[c] = 0
                    else:
                        tmpgrid[a] = tmpgrid[b]
                        tmpgrid[b] = 0
                elif canjoin(tmpgrid, a, b):
                    tmpgrid[a] = tmpgrid[a] + tmpgrid[b]
                    tmpgrid[b] = 0


        self.lasttile = (self.lasttile + 1) % 3
        # TODO: fill random element
        return tmpgrid





    def grid(self):
        # without this the grid may be messed up
        time.sleep(0.2)

        tmp = np.array(self.driver.execute_script("return game.grid;"))
        # tmp = np.log(tmp + 1)
        # tmp = tmp / np.max(tmp)

        self.new = not np.array_equal(tmp, self.lastgrid)

        self.lastgrid = tmp

    def replot(self):
        self.i += 1
        self.line.set_data(range(len(self.scores)), self.scores)

        if len(self.scores) % self.average_over == 0:
            averages = np.mean(
                np.array(self.average_over * [0] + self.scores).reshape(
                    -1, self.average_over),
                axis=1)
            self.avgline.set_data(
                self.average_over * np.array(range(len(averages))), averages)

        self.ax.set_xlim(0, len(self.scores))
        self.ax.set_ylim(0, max(self.scores))
        self.fig.canvas.draw()

    def reset(self):
        self.scores.append(self.lastscore)
        self.replot()
        self.driver.execute_script("game.newGame();")
        self.lastgrid = 16 * [0]
        self.lastscore = 0
        self.nr_of_invalid_moves = 0
        self.grid()

        # replace_function = """
        # ThreesGame.prototype.plantFood = function(t) {
        # var e = void 0
        # , o = void 0
        # , n = void 0
        # , i = void 0
        # , r = [];
        # t[0] ? (e = t[0] > 0 ? this.gridSize - 1 : 0,
        # n = e,
        # o = 0,
        # i = this.gridSize - 1) : (o = t[1] > 0 ? this.gridSize - 1 : 0,
        # i = o,
        # e = 0,
        # n = this.gridSize - 1);
        # for (var s = e; s <= n; s++)
        # for (var a = o; a <= i; a++) {
        # var u = [s, a];
        # this.getGrid(u) || r.push(u)
        # }
        # var c = r[Math.floor(0 * r.length)];
        # this.setGrid(c, this.food),
        # this.animations.push({
        # type: "new",
        # num: this.food,
        # where: c
        # }),
        # this.food = 1 == this.food ? 2 : 1;
        # }
        # """
        # self.driver.execute_script(replace_function)
        if self.square_grid:
            return np.array(self.lastgrid).reshape(-1, 4)
        return np.array(self.lastgrid)

    def done(self):
        return 0 == self.driver.execute_script("return game.canMove.size;")

    def score(self):
        newscore = self.driver.execute_script("return game.score;")
        # s = 1
        # s = max(self.lastgrid)
        # s = newscore - self.lastscore
        # s = newscore / max(1, self.lastscore)
        s = newscore
        s += np.sum(self.lastgrid == 0)# - max([np.sum(self.lastgrid == i) for i in self.lastgrid])

        if newscore == self.lastscore:
            s = 0

        # s += 1
        self.lastscore = newscore
        return s

    def seed(self, xy):
        pass

    def step(self, key):
        self.driver.find_element_by_xpath('//body').send_keys(
            self.action_space.actions[key])
        self.grid()
        reward = self.score()

        if not self.new:
            self.nr_of_invalid_moves += 1
        else:
            self.nr_of_invalid_moves = 0

        done = self.done() or (self.stop_on_invalid_move and
                               self.nr_of_invalid_moves > self.max_invalid_moves)

        observation = np.array(self.lastgrid)
        if self.square_grid:
            observation = np.array(self.lastgrid).reshape(-1, 4)
        return (observation, reward, done, {})

    def render(self):
        pass

    def __del__(self):
        self.driver.close()
