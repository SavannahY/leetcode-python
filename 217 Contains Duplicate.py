//https://leetcode.com/problems/contains-duplicate/
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        m= defaultdict(int)
        
        for i in nums:
            if(m[i]):
                return True
            m[i] += 1
        
        return False
