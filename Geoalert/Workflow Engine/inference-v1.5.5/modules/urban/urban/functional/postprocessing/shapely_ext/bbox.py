class BBox:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    @property
    def xs(self):
        return self.x1, self.x2

    @property
    def ys(self):
        return self.y1, self.y2

    def left_top(self):
        return min(self.xs), max(self.ys), max(self.xs), min(self.ys)

    def left_bot(self):
        return min(self.xs), min(self.ys), max(self.xs), max(self.ys)

    def swne(self):
        return min(self.ys), min(self.xs), max(self.ys), max(self.xs)

    @property
    def geometry(self):
        left, bottom, right, top = self.left_bot()
        geometry = {
            'type': 'Polygon',
            'coordinates': [[[left, top],
                             [right, top],
                             [right, bottom],
                             [left, bottom],
                             [left, top]]]
        }
        return geometry
