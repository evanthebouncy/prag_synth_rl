# some awful global variables indicating the max size of the world
MAX_LEN = 10

# a rectangle is parameterized by Top/Bottom/Left/Right

class Rect:

    # initialize a rectangle with top, bottom, left, right
    def __init__(self, T,B,L,R) -> None:
        self.T = T
        self.B = B
        self.L = L
        self.R = R

    def __str__(self) -> str:
        return "R({},{},{},{})".format(self.T, self.B, self.L, self.R)

    def __repr__(self) -> str:
        return "R({},{},{},{})".format(self.T, self.B, self.L, self.R)

    # check if a point is inside the rectangle
    def is_inside(self, x, y) -> bool:
        return self.T <= y <= self.B and self.L <= x <= self.R

    # turn it into a function that returns a boolean using the is_inside function
    def __call__(self, x, y) -> bool:
        return self.is_inside(x, y)

    # given a list of (x,y),bool pairs check if the rectangle is consistent with the points
    def consistent(self, point_bools: list) -> bool:
        for (x,y),b in point_bools:
            if not self(x,y) == b:
                return False
        return True

    # generate a png image of the rectangle
    def draw(self, filename: str, examples = []) -> None:
        import matplotlib.pyplot as plt
        # set the boundaries of the plot to be 10 by 10
        plt.xlim(0, MAX_LEN)
        plt.ylim(0, MAX_LEN)
        # inverse the y axis
        plt.gca().invert_yaxis()
        # show the x axis labels from 1 to 10
        plt.xticks(range(0, MAX_LEN))
        # show the y axis labels from 1 to 10
        plt.yticks(range(0, MAX_LEN))
        # draw the rectangle with thick borders
        # draw the top line from L,T to R,T
        plt.plot([self.L, self.R], [self.T, self.T], 'k-', linewidth=2)
        # draw the bottom line from L,B to R,B
        plt.plot([self.L, self.R], [self.B, self.B], 'k-', linewidth=2)
        # draw the left line from L,T to L,B
        plt.plot([self.L, self.L], [self.T, self.B], 'k-', linewidth=2)
        # draw the right line from R,T to R,B
        plt.plot([self.R, self.R], [self.T, self.B], 'k-', linewidth=2)

        # draw the examples
        for ex in examples:
            # if the ex does not contain a boolean
            if type(ex[1]) != bool:
                if self.is_inside(ex[0], ex[1]):
                    # draw a big green dot
                    plt.plot(ex[0], ex[1], 'go', markersize=10)
                else:
                    plt.plot(ex[0], ex[1], 'ro', markersize=10)
            else:
                xy, b = ex
                if b:
                    plt.plot(xy[0], xy[1], 'go', markersize=10)
                else:
                    plt.plot(xy[0], xy[1], 'ro', markersize=10)

        plt.savefig(filename)
        plt.close()

if __name__ == '__main__':
    rect = Rect(1,6,1,6)
    print(rect.is_inside(5,5))
    print(rect.is_inside(6,6))
    print(rect.is_inside(7,7))
    print(Rect(1,6,1,6)(5,6))

    Rect(1,3,4,9).draw('tmp/rect.png', [(1,1), (2,2), (3,3), (4,4), (4,1), (5,2), (9,3)])

    test_coords = []
    for x in range(0, MAX_LEN+1):
        for y in range(0, MAX_LEN+1):
            test_coords.append((x,y))
    examples = [((x,y),Rect(1,3,4,9)(x,y)) for x,y in test_coords]
    Rect(1,3,4,9).draw('tmp/rect2.png', examples)