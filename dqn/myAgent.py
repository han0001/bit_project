import tensorflow as tf
import numpy as np
import random
import time
import cv2

import myGame
import myModel

# 최대 학습 횟수
MAX_EPISODE = 500000
# 1000번의 학습마다 한 번씩 타겟 네트웍을 업데이트합니다.
TARGET_UPDATE_INTERVAL = 20000
# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4
# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
OBSERVE = 5000
EPSILON_DIV = 2000000 # EPSILON_DIV step이후 만큼 DQN의 action수행

# action: 1직진, 2좌, 3우
NUM_ACTION = 4
SHOW_GAME = True
IMG_NAME = 'testmap5.png'

def loadMap():
    img = cv2.imread(IMG_NAME, cv2.IMREAD_GRAYSCALE)
    img = np.where(img<128,1,img)
    img = np.where(img>128,0,img)

    return img

def train():
    #초기세팅
    mapData=loadMap()
    rbInitPt = [1, 1]

    print("train 시작")
    sess = tf.Session()

    game = myGame.Game(mapData, rbInitPt, show_game=SHOW_GAME)
    brain = myModel.DQN(sess, mapData.shape[0], mapData.shape[1], NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    # 타겟 네트웍을 초기화합니다.
    brain.update_target_network()

    # 다음에 취할 액션을 DQN 을 이용해 결정할 시기를 결정합니다.
    epsilon = 1.0
    # 프레임 횟수
    time_step = 0
    total_reward_list = []
    reward_temp = -1000

    # 게임을 시작합니다.
    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        # 게임을 초기화하고 현재 상태를 가져옵니다.
        # 상태는 screen_width x screen_height 크기의 화면 구성입니다.
        state = game.reset()
        brain.init_state(state)

        while not terminal:
            # 입실론이 랜덤값보다 작은 경우에는 랜덤한 액션을 선택하고
            # 그 이상일 경우에는 DQN을 이용해 액션을 선택합니다.
            # 초반엔 학습이 적게 되어 있기 때문입니다.
            # 초반에는 거의 대부분 랜덤값을 사용하다가 점점 줄어들어
            # 나중에는 거의 사용하지 않게됩니다.
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            # 일정 시간이 지난 뒤 부터 입실론 값을 줄입니다.
            # 초반에는 학습이 전혀 안되어 있기 때문입니다.
            if episode > OBSERVE:
                epsilon -= 1 / EPSILON_DIV
            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal = game.step(action)
            total_reward += reward

            # 현재 상태를 Brain에 기억시킵니다.
            # 기억한 상태를 이용해 학습하고, 다음 상태에서 취할 행동을 결정합니다.
            brain.remember(state, action, reward, terminal)

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                # DQN 으로 학습을 진행합니다.
                brain.train()

            if time_step % TARGET_UPDATE_INTERVAL == 0:
                # 타겟 네트웍을 업데이트 해 줍니다.
                brain.update_target_network()

            time_step += 1

        print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        # if episode % 10 == 0:
        #     summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
        #     writer.add_summary(summary, time_step)
        #     total_reward_list = []

        if reward_temp < total_reward:
            saver.save(sess, 'mymodel/dqn.ckpt', global_step=time_step)
            reward_temp = total_reward
            f = open("mymodel/dqn.ckpt-" + str(time_step) + "hypram.txt", "w")
            f.write("IMG_NAME : " + IMG_NAME + "\n\n" +
                    "-Agent-\n"+
                    "  MAX_EPISODE : " + str(MAX_EPISODE)+ "\n"+
                    "  TARGET_UPDATE_INTERVAL : " + str(TARGET_UPDATE_INTERVAL)+ "\n"+
                    "  TRAIN_INTERVAL : "+ str(TRAIN_INTERVAL)+ "\n"+
                    "  OBSERVE : "+ str(OBSERVE)+ "\n"+
                    "  EPSILON_DIV : "+ str(EPSILON_DIV)+ "\n\n"+
                    "-Model-\n"+
                    "  REPLAY_MEMORY : "+str(brain.REPLAY_MEMORY)+ "\n"+
                    "  BATCH_SIZE : "+ str(brain.BATCH_SIZE)+ "\n"+
                    "  GAMMA : "+ str(brain.GAMMA)+ "\n"+
                    "  STATE_LEN : "+ str(brain.STATE_LEN)+ "\n"+
                    "-Model-\n" +
                    "  limit_step : "+ str(game.limit_step)+ "\n\n"+
                    "-----record----\n"+
                    "  episode : "+ str(episode+1)+ "\n"+
                    "  iswin : "+ str(game.iswin)+ "\n"+
                    "  total_reward : "+ str(total_reward)+ "\n"
                    "  step_cnt : "+ str(game.step_cnt)+ "\n"
                    "  epsilon : "+ str(epsilon)+ "\n"
                    "  all_action : "+ str(game.all_action)+ "\n"
                    )
            f.close()

            f = open("mymodel/dqn.ckpt-" + str(time_step) + "action.txt", "w")
            f.write(str(game.all_action))
            f.close()

def replay():
    mapData=loadMap()
    rbInitPt = [1, 1]

    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = myGame.Game(mapData, rbInitPt, show_game=True)
    game.reset()

    line = []
    f = open("mymodel/dqn.ckpt-2026187action.txt", 'r')
    line = f.readline()
    f.close()

    line = line.replace("[", "")
    line = line.replace("]", "")
    line = line.replace(",", "")
    line = line.replace(" ", "")

    line = list(line)

    for ii in range(10):
        for i in line:
            game.step(int(i))
        game.reset()

    a = input()

# def replay():
#     mapData=loadMap()
#     rbInitPt = [1, 1]
#
#     print('뇌세포 깨우는 중..')
#     sess = tf.Session()
#
#     game = myGame.Game(mapData, rbInitPt, show_game=True)
#     brain = myModel.DQN(sess, mapData.shape[0], mapData.shape[1], NUM_ACTION)
#
#     saver = tf.train.Saver()
#     ckpt = tf.train.get_checkpoint_state('mymodel')
#     saver.restore(sess, ckpt.model_checkpoint_path)
#
#     # 게임을 시작합니다.
#     for episode in range(MAX_EPISODE):
#         terminal = False
#         total_reward = 0
#
#         state = game.reset() #여기 괄호 가끔 사라짐.. 왜지?
#         brain.init_state(state)
#
#         while not terminal:
#             action = brain.get_action()
#
#             # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
#             state, reward, terminal = game.step(action)
#             total_reward += reward
#
#             brain.remember(state, action, reward, terminal)
#
#             # 게임 진행을 인간이 인지할 수 있는 속도로^^; 보여줍니다.
#             time.sleep(0.05)
#
#         print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))





# train()
replay()