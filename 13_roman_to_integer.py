#
# @lc app=leetcode id=13 lang=python3
#
# [13] Roman to Integer
#

# @lc code=start
class Solution:
    def romanToInt(self, s: str) -> int:
        r = {'I':1, 'V':5, 'X':10,'L':50, 'C':100,'D':500,'M':1000}
        total = 0
        for i in range(len(s)-1):
            if r[s[i]] < r[s[i+1]]:
                total -= r[s[i]]
            else:
                total += r[s[i]]
        total += r[s[-1]]
        return total
