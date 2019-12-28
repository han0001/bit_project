# 청소가능 영역 0
# 장애물 1
# 청소된 영역 2
# 로봇위치 3

import copy
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time



class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class Game:
    def __init__(self, mapData, rbInitPt, show_game=True):
        # 초기화면 설정
        self.mapData = copy.deepcopy(mapData)
        self.state = copy.deepcopy(mapData)
        self.screen_width = mapData.shape[1]
        self.screen_height = mapData.shape[0]
        self.gameover = 0

        self.rbInitPt = Position(rbInitPt[0], rbInitPt[1])  # 로봇 초기 좌표
        self.robot = Position(1, 1)  # 로봇이 움직일 좌표

        self.show_game = show_game  # 게임 화면 on/off

        self.blocks = set()  # 장애물 지도에서 받아서 set에 넣기
        for row in range(self.screen_height):
            for col in range(self.screen_width):
                if self.mapData[row][col] == 1:
                    self.blocks.add(Position(col, row))

        self.cleandAreas = set()  # 청소한 영역

        self.one_hot_action = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        self.step_cnt = 0
        self.limit_step = int(self.screen_width * self.screen_height * 1.4)
        # self.limit_step = 10
        self.cleand_cnt = np.count_nonzero(mapData == 0)  # 청소할 공간개수
        self.all_action = []
        self.iswin = 0
        # if show_game:
        #     self.fig, self.axis = self._prepare_display()

    def _prepare_display(self):
        # 해상도 설정을 위한 화면 크기 계산
        set_width = int((self.screen_width / (self.screen_height + self.screen_width) * 15))
        set_height = int((self.screen_height / (self.screen_height + self.screen_width) * 15))

        fig, axis = plt.subplots(figsize=(set_width, set_height))  # figsize 8x8 = 800x800
        fig.set_size_inches(set_width, set_height)
        plt.axis((0, self.screen_width, 0, self.screen_height))  # 눈금표시

        plt.draw()
        plt.ion()
        plt.gca().invert_yaxis()
        plt.show()

        return fig, axis

    def _get_state(self):
        return self.state

    def _draw_screen(self):
        grayimage = self._get_state()
        rgbimage = np.stack((grayimage,) * 3, axis=-1)

        rgbimage = np.where(rgbimage == [0, 0, 0], [255, 255, 255], rgbimage)
        rgbimage = np.where(rgbimage == [1, 1, 1], [0, 0, 0], rgbimage)
        rgbimage = np.where(rgbimage == [2, 2, 2], [190, 219, 0], rgbimage)
        rgbimage = np.where(rgbimage == [3, 3, 3], [0, 0, 170], rgbimage)
        rgbimage = rgbimage.astype(np.uint8)

        title1 = " Reward: %d Step : %d " % (
            self.current_reward,
            self.step_cnt)

        title2 = " Total Game: %d " % (
            148763)

        # title1 = " Reward: %d Step : %d " % (
        #     self.current_reward,
        #     self.step_cnt)
        #
        # title2 = " Total Game: %d " % (
        #     self.total_game)

        img_w = 300
        img_h = 100
        bpp = 3

        img = np.zeros((img_h, img_w, bpp), np.uint8)

        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        cyan = (255, 255, 0)
        magenta = (255, 0, 255)

        center_x = int(img_w / 2.0)
        center_y = int(img_h / 2.0)

        thickness = 2

        location = (0, 30)
        font = cv2.FONT_HERSHEY_DUPLEX  # hand-writing style font
        fontScale = 0.7
        cv2.putText(img, title1, location, font, fontScale, white, thickness)

        location = (0, 70)
        font = cv2.FONT_HERSHEY_DUPLEX  # hand-writing style font
        fontScale = 0.7
        cv2.putText(img, title2, location, font, fontScale, white, thickness)

        cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
        cv2.imshow("drawing", img)
        cv2.imshow('Resized Window', rgbimage)
        cv2.waitKey(100)
        print(self._get_state())

    # def _draw_screen(self):
    #     title = "# Avg. Reward: %d Reward: %d Total Game: %d Step : %d #" % (
    #         self.total_reward / self.total_game,
    #         self.current_reward,
    #         self.total_game,
    #         self.step_cnt)
    #
    #
    #     self.axis.set_title(title, fontsize=10)
    #
    #     # 블럭 그리기
    #     for block in self.blocks:
    #         block0 = patches.Rectangle((block.x, block.y),
    #                                    1, 1,
    #                                    linewidth=0, facecolor="#000000")  # plt그래프에 넣기전 설정
    #         self.axis.add_patch(block0)  # plt그래프에 추가하는 함수
    #
    #     # 청소된 영역 그리기
    #     for cleanArea in self.cleandAreas:
    #         cleanArea0 = patches.Rectangle((cleanArea.x, cleanArea.y),
    #                                        1, 1,
    #                                        linewidth=0, facecolor="#A9F5F2")  # plt그래프에 넣기전 설정
    #         self.axis.add_patch(cleanArea0)  # plt그래프에 추가하는 함수
    #
    #     # 로봇 그리기
    #     robot = patches.Rectangle((self.robot.x, self.robot.y),
    #                               1, 1,
    #                               linewidth=0, facecolor="#B40431")
    #     self.axis.add_patch(robot)
    #
    #     self.fig.canvas.draw()
    #     # 게임의 다음 단계 진행을 위해 matplot 의 이벤트 루프를 잠시 멈춥니다.
    #     plt.pause(0.0001)

    def reset(self):
        # 자동차, 장애물의 위치와 보상값들 초기화
        self.state = copy.deepcopy(self.mapData)
        self.current_reward = 0
        self.total_game += 1
        self.all_action.clear()
        self.gameover = 0

        self.robot = Position(1, 1)
        self.cleandAreas.clear()
        self.step_cnt = 0

        return self._get_state()

    def _update_robot(self, move):
        #청소된구역 넣기
        self._update_clean_area(self.robot)
        self.state[self.robot.y][self.robot.x] = 2

        self.robot.x = self.robot.x + self.one_hot_action[move][0]
        self.robot.y = self.robot.y + self.one_hot_action[move][1]

        reward = 1
        #위0 아래1 좌2 우3
        if self.robot in self.blocks: #게임 오버
            self.gameover = 1
            self.total_reward += self.current_reward
            reward -= 3

        elif self.robot in self.cleandAreas: #감점
            reward -= 2

        self.state[self.robot.y][self.robot.x] = 3

        return reward

    def _update_clean_area(self, bf_robot):
        if bf_robot not in self.cleandAreas:
            self.cleandAreas.add(copy.deepcopy(bf_robot))



    def step(self, action):
        self.step_cnt += 1
        self.all_action.append(action)

        # action 1위 ,2아래 ,3좌 ,4우
        reward = self._update_robot(action)
        if self.cleand_cnt <= len(self.cleandAreas):
            reward += 5
            self.iswin = 1
            print("################################깻다")

        if self.limit_step < self.step_cnt or self.current_reward < -100:
            self.gameover = True

        self.current_reward += reward
        # print(self.limit_step, self.step_cnt, len(self.cleandAreas))


        if self.show_game:
            self._draw_screen()

        return self._get_state(), reward, self.gameover


#
# # ------------------------ 테스트용 -------------------
# import cv2
#
# def loadMap():
#     img = cv2.imread('testmap3.png', cv2.IMREAD_GRAYSCALE)
#     img = np.where(img<128,1,img)
#     img = np.where(img>128,0,img)
#
#     return img
#
# def test():
#     #초기세팅
#     mapData=loadMap()
#     rbInitPt = [1, 1]
#
#     game = Game(mapData, rbInitPt, show_game=True)
#     game.reset()
#     for i in range(100):
#         s = input()
#         game.step(int(s))
#
# test()