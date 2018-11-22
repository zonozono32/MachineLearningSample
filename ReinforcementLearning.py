import random
import numpy as np

# 迷路

MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0],
    [0, 1, 1, 3, 3, 3, 3, 3, 1, 1, 0],
    [0, 1, 1, 3, 3, 3, 3, 3, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

MAX_X = len(MAP[0])
MAX_Y = len(MAP)

MAX_EPI = 2000
MAX_STEP = 200
ACT_NUM = 4

NEXT = [
    [-1, 0],
    [ 1, 0],
    [ 0,-1],
    [ 0, 1]
]

ARROW = ["↑", "↓", "←", "→"]
MAP_ICON = ["■", "□", "Ｇ", "Ｘ"]

sx = 1
sy = 1

# TD学習用

ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.3
Q = np.random.rand(MAX_Y, MAX_X, ACT_NUM) - 0.5

def action(x, y):
    if random.random() < EPSILON:
        return random.randrange(ACT_NUM)
    else:
        return np.argmax(Q[y][x])

def sarsa(x, y, a, r, nx, ny, na):
    Q[y][x][a] = Q[y][x][a] + ALPHA * (r + GAMMA * Q[ny][nx][na] - Q[y][x][a])

def showArrow():
    for y, items in enumerate(Q):
        str = ""
        for x, item in enumerate(items):
            if MAP[y][x] == 1:
                act = np.argmax(item)
                str += ARROW[act]
            else:
                str += MAP_ICON[MAP[y][x]]
        print(str)


for epi in range(MAX_EPI):
    x = sx
    y = sy

    nx = sx
    ny = sy

    na = 0

    a = action(x, y)

    for step in range(MAX_STEP):
        ty = y + NEXT[a][0]
        tx = x + NEXT[a][1]

        if MAP[ty][tx] != 0:
            ny += NEXT[a][0]
            nx += NEXT[a][1]

        na = action(x, y)

        if MAP[y][x] == 2:
            sarsa(x, y, a, 40, nx, ny, na)
            break
        elif MAP[y][x] in {0, 3}:
            sarsa(x, y, a, -20, nx, ny, na)
            break
        else:
            sarsa(x, y, a, -1, nx, ny, na)

        x = nx
        y = ny
        a = na

    print("episode", epi + 1)
    showArrow()
