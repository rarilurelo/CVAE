

class hoge():
    def __init__(self, x):
        self.x = x


    def __call__(self, x):
        return self.geo(x)
    def geo(self, x):
        return x+1


v = hoge(10)(9)

