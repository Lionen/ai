import json


class State:
    Added = 1
    Updated = 2
    Deleted = 3


class Intent:
    Left = 1
    Right = 2


class Box:
    # def __init__(self, id, feature):
    #     self.id = id
    #     self.feature = feature
    #     self.bboxes = []
    #     self.state = State.Added

    def __init__(self, id, feature, bboxes):
        self.id = id
        self.feature = feature
        self.bboxes = bboxes
        # self.state = state
        self.intent = None

    def update(self, feature, bbox):
        self.feature = feature

        x = bbox[0] - self.bboxes[-1][0]
        if x > 0:
            self.intent = Intent.Right
        elif x < 0:
            self.intent = Intent.Left

        if len(self.bboxes) == 2:
            self.bboxes[-1] = bbox
        else:
            self.bboxes.append(bbox)

    # def delete(self):
    #     self.state = State.Deleted

    def is_left(self):
        return self.intent == Intent.Left

    def is_right(self):
        return self.intent == Intent.Right


def handle(d):
    return Box(d['id'], d['feature'], d['bboxes'])
