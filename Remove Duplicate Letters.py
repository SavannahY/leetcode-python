class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        dic = {}
        for c in s:
            dic[c] = dic.get(c,0)+1
        res =[]
        visited = set()
        for c in s:
            dic[c] -= 1
            if c not in visited:
                while res and c < res[-1] and dic[res[-1]]>0:
                    visited.remove(res.pop())
                res.append(c)
                visited.add(c)
        return("".join(res))
