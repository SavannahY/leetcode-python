#
# @lc app=leetcode id=881 lang=python3
#
# [881] Boats to Save People
#

# @lc code=start
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        i = 0
        j = len(people) - 1
        ans = 0
        while(i <= j):
            if i == j:
                ans += 1
                return ans
            elif(people[i] + people[j] > limit):
                j -= 1
                ans += 1
            else:
                j -= 1
                i += 1
                ans += 1

        return ans
