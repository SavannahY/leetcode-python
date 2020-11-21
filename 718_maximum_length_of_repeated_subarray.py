#
# @lc app=leetcode id=718 lang=python3
#
# [718] Maximum Length of Repeated Subarray
#

# @lc code=start
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        r = len(A) + 1
        c = len(B) + 1
        if r == 1 or c == 1:
            return 0
        res = 0
        dp = [[0]* c for _ in range(r)]
        for i in range(1,r):
            for j in range(1,c):
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[i-1][j-1] +1
                    res = max(res,dp[i][j])
        return res
