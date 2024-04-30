from enum import IntEnum

class StaticItem(IntEnum):
    EMPTY = 0
    WALL = 1
    POT = 2
    DISH_DISPENSER = 3

def get_ingredient_dispenser_encoding(ingredient):
    return StaticItem.DISH_DISPENSER + ingredient


class DynamicItem(IntEnum):
    PLATE = 1 << 0
    ONION = 1 << 1
    TOMATO = 4
    LETTUCE = 4
    DISH = 5


def get_encoding(maze_map):
    pass
