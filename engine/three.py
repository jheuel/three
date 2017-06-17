#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PoorMansGymEnv(object):
    class ActionSpace(object):
        def __init__(self):
            self.directions = [Keys.LEFT, Keys.RIGHT, Keys.DOWN, Keys.UP]
            self.actions = [Keys.LEFT, Keys.RIGHT, Keys.DOWN]#, Keys.UP]
            self.n = len(self.actions)

    def __init__(self, mocked_game=False, stop_on_invalid_move=False, average_over=50, max_invalid_moves=5, square_grid=False):
        self.square_grid = square_grid
        self.stop_on_invalid_move = stop_on_invalid_move
        self.max_invalid_moves = max_invalid_moves
        self.scores = []
        self.i = 0
        self.lastscore = 0
        self.action_space = self.ActionSpace()

        self.mocked_game = mocked_game
        if not self.mocked_game:
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
            self.fig.savefig('/home/jheuel/Dropbox/three.pdf')

        self.ax.set_xlim(0, len(self.scores))
        self.ax.set_ylim(0, max(self.scores))
        self.fig.canvas.draw()

    def reset(self):
        self.scores.append(self.lastscore)
        self.replot()
        if not self.mocked_game:
            self.driver.execute_script("game.newGame();")
        self.lastgrid = 16 * [0]
        if self.mocked_game:
            self.lastgrid[np.random.randint(0, len(self.lastgrid))] = 1
            self.lastgrid[np.random.randint(0, len(self.lastgrid))] = 2
        self.lastscore = 0
        self.nr_of_invalid_moves = 0

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
        if not self.mocked_game:
            return 0 == self.driver.execute_script("return game.canMove.size;")
        else:
            done = True
            tmpgrid = np.array(self.lastgrid).reshape(-1, 4)
            for i in range(4):
                for j in range(4):
                    if i+1 < 4 and self.canjoin(tmpgrid, (i,j), (i+1,j)):
                        return False
                    if i-1 > -1 and self.canjoin(tmpgrid, (i,j), (i-1,j)):
                        return False
                    if j+1 < 4 and self.canjoin(tmpgrid, (i,j), (i,j+1)):
                        return False
                    if j-1 > -1 and self.canjoin(tmpgrid, (i,j), (i,j-1)):
                        return False
            return done

    def score(self):
        if not self.mocked_game:
            self.lastscore = self.driver.execute_script("return game.score;")

        return self.lastscore

    def seed(self, xy):
        pass

    def step(self, key):
        if not self.mocked_game:
            self.driver.find_element_by_xpath('//body').send_keys(
                self.action_space.actions[key])

            # without this the grid may be messed up
            time.sleep(0.2)
            tmp = np.array(self.driver.execute_script("return game.grid;"))
        else:
            self.simulate(key)
            tmp = self.newgrid

        merged_tiles_last_move = np.sum(np.array(self.lastgrid) > 0) - np.sum(np.array(tmp) > 0)
        self.new = not np.array_equal(tmp, self.lastgrid)
        self.lastgrid = tmp

        reward = merged_tiles_last_move + 0.1

        if not self.new:
            self.nr_of_invalid_moves += 1
        else:
            self.nr_of_invalid_moves = 0

        done = self.done() or (self.stop_on_invalid_move and
                               self.nr_of_invalid_moves > self.max_invalid_moves)

        observation = np.array(self.lastgrid)
        if self.square_grid:
            observation = np.array(self.lastgrid).reshape(-1, 4)

        observation = np.log(observation + 1)
        # observation = observation / np.max(observation)
        return (observation, reward, done, {})

    def canjoin(self, grid, a, b):
        return grid[a] < 3 and grid[b] < 3 and grid[a] != grid[b] or grid[a] > 2 and grid[a] == grid[b]

    def simulate(self, direction):
        key = self.action_space.directions[direction]
        tmpgrid = np.array(self.lastgrid).reshape(-1, 4)
        x = {
                Keys.LEFT:  lambda i,j: ((i,j), (i,j+1), (i,j+2)),
                Keys.RIGHT: lambda i,j: ((i,3-j), (i,3-(j+1)), (i,3-(j+2))),
                Keys.UP:    lambda j,i: ((i,j), (i+1,j), (i+2,j)),
                Keys.DOWN:  lambda j,i: ((3-i,j), (3-(i+1),j), (3-(i+2),j)),
        }

        for i in range(4):
            for j in range(3):
                a, b, c = x[key](i,j)
                if tmpgrid[a] == 0:
                    if max(c) < 4 and min(c) >= 0 and tmpgrid[b] > 0 and self.canjoin(tmpgrid, b, c):
                        two_tile_sum = tmpgrid[b] + tmpgrid[c]
                        tmpgrid[a] = two_tile_sum
                        self.lastscore += two_tile_sum
                        tmpgrid[b] = 0
                        tmpgrid[c] = 0
                    else:
                        tmpgrid[a] = tmpgrid[b]
                        tmpgrid[b] = 0
                elif self.canjoin(tmpgrid, a, b):
                    two_tile_sum = tmpgrid[a] + tmpgrid[b]
                    tmpgrid[a] = two_tile_sum
                    self.lastscore += two_tile_sum
                    tmpgrid[b] = 0


        self.lasttile = (self.lasttile + 1) % 2

        tmpgrid = np.ndarray.flatten(tmpgrid)
        indices = np.array(range(len(tmpgrid)))[tmpgrid == 0]
        if len(indices) > 0:
            tmpgrid[np.random.choice(indices)] = self.lasttile+1

        self.newgrid = tmpgrid

    def __del__(self):
        if not self.mocked_game:
            self.driver.close()
