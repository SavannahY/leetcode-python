class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.st = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        curMin = self.getMin()
        if curMin is None or x < curMin:
            curMin = x
        self.st.append([x,curMin])
        

    def pop(self):
        """
        :rtype: None
        """
        self.st.pop()
        

    def top(self):
        """
        :rtype: int
        """
        return self.st[-1][0] if self.st else None
        

    def getMin(self):
        """
        :rtype: int
        """
        return self.st[-1][1] if self.st else None
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
