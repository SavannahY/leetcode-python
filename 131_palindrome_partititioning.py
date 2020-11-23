#
# @lc app=leetcode id=131 lang=python3
#
# [131] Palindrome Partitioning
#

# @lc code=start
class Solution:
    def isPalin(self, s):
        """
        test if the segement string is palindrome
        """
        i = 0
        j = len(s)-1
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
    def dfs(self, s:str):
        """
        recurcively find and append temp list to res
        """
        if(len(s) == 0 and len(self.temp) > 0):
            self.res.append(self.temp[:])
            return
        n = len(s)+ 1
        for i in range(1,n):
            seg = s[0:i]
            if(self.isPalin(seg)):
                self.temp.append(seg)
                self.dfs(s[i:])
                self.temp.pop()
    def partition(self, s: str) -> List[List[str]]:
        self.res = []
        self.temp = []
        self.dfs(s)
        return self.res
        
# @lc code=end
