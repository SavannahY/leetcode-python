## 入门

### 二分法 array基本

#### 35. Search Insert Position

nums contains distinct values sorted in ascending order

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if target in nums:
            return nums.index(target)
        else:
            nums.append(target)
            nums.sort()
            return nums.index(target)
```

pure binary search:

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1 
            else:
                r = mid - 1
        if l > r:
            return r + 1
        else:
            return l
```

1. A[mid] 不是 mid

2. while l <= r 不是 while l < r

   

#### 704. Binary Search

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
		return nums.index(target) if target in nums else -1
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return -1
        
```



l <= r 不是 1 <= r

third bisect

```python
class Solution:
    def search(self, nums, target):
        index = bisect.bisect_left(nums, target)
        return index if index < len(nums) and nums[index] == target else -1
```

#### 852. Peak Index in a Mountain Array

```
l, r = 0, len(A)-1
while True: # loops forever since there will be always be a peak in the test case:
            mid = (l + r) // 2
            if A[i-1] < A[i] > A[i+1]:
                return mid
            elif A[i-1] < A[i] < A[i+1]:
                l = mid + 1
            else:
                r = mid - 1
```

#### 349. Intersection of Two Arrays

Given two arrays, write a function to compute their intersection.

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))
```



#### 167. Two Sum two - input array is sorted

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        investigated = []
        for i in range(len(numbers)):
            if not numbers[i] in investigated:
                investigated.append(numbers[i])
                l, r = i + 1, len(numbers) - 1
                tmp = target - numbers[i]
                while l <= r:
                    mid = l + (r -l) // 2
                    if numbers[mid] == tmp:
                        return([i + 1, mid + 1])
                    elif numbers[mid] < tmp:
                        l = mid + 1
                    else:
                        r = mid - 1
```



#### 81. Search in Rotated Sorted Array two

#### 153. Find Minimum in Rotated Sorted Array

#### 981. Time Based Key-Value Store

#### 287. ind the Duplicate Number

#### 209. Minimum Size Subarray Sum

#### 300. Longest Increasing Subsequence

#### 378. Kth Smallest Element in a Sorted Matrix

#### 74. Search a 2D matrix

### 各种各样的Array

#### Longest Harmonious Subsequence

最长的字串。字串的最大值和最小值相差1

排序 

count 

Item item count

相邻两个 的count 和 同时还是值差1 的记录

记录数据

返回数据* count 

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        counter = collections.Counter(nums)
        ans = 0
        for num in counter:
            if num+1 in counter:
                ans = max(ans, counter[num] + counter[num+1])
        return ans
                
```

1. Collections 有s
2. Counter[x] 是[] 不是() 
3. max很好用

Counter是一个frenquency table

```
        freq = {}
        for x in nums: freq[x] = 1 + freq.get(x, 0)
```

### Sorting Algorithms

#### 912. Sort an array

sorted[nums]

1. *Selection Sort:*
   This program sorts the list by finding the minimum of the list, removing it from the original list, and appending it onto the output list. As it finds the minumum during each iteration, the length of the original list gets shorter by one and the length of the output list gets longer by one. This program uses the min function with a lambda function to find the index of the minimum value in the unsorted list. The output list is created using a list comprehension. This algorithm is O(n²). It is able to sort 8 out of 10 test cases but exceeds the time limit on the ninth test case.

```
class Solution:
    def sortArray(self, N: List[int]) -> List[int]:
        L = len(N)
        return [N.pop(min(range(L-i), key = lambda x: N[x])) for i in range(L)]
		
```

1. *Bubble Sort:*
   This program sorts the list by looping through the original list and finding any adjacent pairs of numbers that are out of order. If it finds such a pair, it switches their locations. It continues to check for adjacent pairs that are out of order until it has completed the pass through the list. If it finds a pair, it will set B = 1 which tells the program to pass through the list again and continue switching adjacent values that are out of order. If it can make it through the list without finding any adjacent pairs that are out of order, we can conclude that the list is sorted. The value of B will stay 0 and the outer while loop will end. This algorithm is O(n²). It is able to sort 8 out of 10 test cases but exceeds the time limit on the ninth test case.

```
class Solution:
    def sortArray(self, N: List[int]) -> List[int]:
        L, B = len(N), 1
        while B:
            B = 0
            for i in range(L-1):
                if N[i] > N[i+1]: N[i], N[i+1], B = N[i+1], N[i], 1
        return N
		
		
```

1. *Insertion Sort:*
   This program sorts the list by sequentially inserting each element into the proper location within the sorted front portion of the original list. It starts by inserting the second number into the proper location within the first two numbers. It then inserts the third number into the proper location within the first three numbers. After each iteration, the front portion of the list (i.e. the sorted portion) will grow in length by 1. A number is removed from the back portion of the list (i.e. the unsorted portion) and placed into the appropriate location in the sorted portion of the list. Once we place the last number into the correct location within the sorted numbers, the entire list will have become sorted. This algorithm is O(n²). It is able to sort 9 out of 10 test cases but exceeds the time limit on the last test case.

```
class Solution:
    def sortArray(self, N: List[int]) -> List[int]:
        L = len(N)
        for i in range(1,L):
            for j in range(0,i):
                if N[i] < N[j]:
                    N.insert(j, N.pop(i))
                    break
        return N		
```

1. *Binary Insertion Sort:* (944 ms) (beats ~6%)
   This program sorts the list by sequentially inserting each element into the proper location within the sorted front portion of the original list. It starts by inserting the second number into the proper location within the first two numbers. It then inserts the third number into the proper location within the first three numbers. After each iteration, the front portion of the list (i.e. the sorted portion) will grow in length by 1. A number is removed from the back portion of the list (i.e. the unsorted portion) and placed into the appropriate location in the sorted portion of the list. Once we place the last number into the correct location within the sorted numbers, the entire list will have become sorted. This algorithm is still O(n²). It is not O(n log n) despite using a binary search because the insertion step is time consuming. The advantage of the binary insertion sort versus the generic insertion sort is that this one does a binary search which is O(log i) (for the ith iteration) while the generic one does a linear search which is O(i) (for the ith iteration). Because of this improvement, this program is able to sort all 10 cases although it is not a very fast approach compared to some of the other algorithms.

```
class Solution:
    def sortArray(self, N: List[int]) -> List[int]:
        L = len(N)
        for i in range(1,L): bisect.insort_left(N, N.pop(i), 0, i)
        return N
```

1. *Counting Sort:* (136 ms) (beats ~97%)
   This program starts by creating a dictionary of key-value pairs that records the total count of each element of the original list. The minimum and maximum value of the list is also found. The program then loops through every integer between the min and max value (in order) and appends that number to the output list based on its count within the original list. For example, if the number 3 occurs 4 times in the original list, then the output list will be extended by [3,3,3,3]. The output will be the sorted form of the original list as it consists of an ordered list of the numbers in the original list. This approach is a linear time sorting algorithm that sorts in O(n+k) time, where n is the number of elements and k is the statistical range of the dataset (i.e. max(N) - min(N)). It is able to sort all test cases in only 136 ms.

```
class Solution:
    def sortArray(self, N: List[int]) -> List[int]:
        C, m, M, S = collections.Counter(N), min(N), max(N), []
        for n in range(m,M+1): S.extend([n]*C[n])
        return S
```

1. *QuickSort:* (316 ms) (beats ~54%)
   This program uses a recursive Divide and Conquer approach to sort the original list. The recursive function `quicksort` is called on the original list. A `partition`subroutine then takes the input and pivots around the median position's value (pivot value). Specifically, it uses swaps to place numbers less than the pivot value to the left of the pivot index and numbers greater than the pivot value to the right of the pivot index. At the end of the `partition` subroutine, we are left with the situation that the value at the pivot index is located at what will be its final position in the sorted output. We then do a quicksort on these left and right halves and continue this rescursively until we get a list of length 1 which is already sorted. Once all of the quicksort subroutines are completed, the entire list is in sorted order and the initial quicksort call completes and we can return the sorted list. In the average case, this method is O(n log n). In the worst case, however, it is O(n²). It is able to sort all test cases in 316 ms.

```
class Solution:
    def sortArray(self, N: List[int]) -> List[int]:
        def quicksort(A, I, J):
            if J - I <= 1: return
            p = partition(A, I, J)
            quicksort(A, I, p), quicksort(A, p + 1, J)
        
        def partition(A, I, J):
            A[J-1], A[(I + J - 1)//2], i = A[(I + J - 1)//2], A[J-1], I
            for j in range(I,J):
                if A[j] < A[J-1]: A[i], A[j], i = A[j], A[i], i + 1
            A[J-1], A[i] = A[i], A[J-1]
            return i
        
        quicksort(N,0,len(N))
        return N
		
		
```

1. *Merge Sort:* (324 ms) (beats ~51%)
   This program uses a recursive Divide and Conquer approach to sort the original list. The recursive function `mergesort` is called on the original list. The function divides the list into two halves (a left half and a right half) and calls the mergesort routine recursively on each of them. Once each half is sorted, the `merge` subroutine merges the two lists into one list by appending the lower number from the front of each half into the output list S. If one list is exhausted before the other, the remaining list is appended onto S before S is returned. The `mergesort` function recursively calls on lists that are half the size of the previous list until their length is 1. Such a list is trivially sorted and that list can be returned for `merge` to join with another list. Eventually all the small lists are merged back into larger lists until the final sorted version of the original list is formed. This method is O(n log n). It is able to sort all test cases in 324 ms.

```
class Solution:
    def sortArray(self, N: List[int]) -> List[int]:
        def mergesort(A):
            LA = len(A)
            if LA == 1: return A
            LH, RH = mergesort(A[:LA//2]), mergesort(A[LA//2:])
            return merge(LH,RH)

        def merge(LH, RH):
            LLH, LRH = len(LH), len(RH)
            S, i, j = [], 0, 0
            while i < LLH and j < LRH:
                if LH[i] <= RH[j]: i, _ = i + 1, S.append(LH[i])
                else: j, _ = j + 1, S.append(RH[j])
            return S + (RH[j:] if i == LLH else LH[i:])
        
        return mergesort(N)		
		
		
```

1. *Bucket Sort:* (196 ms) (beats ~78%)
   This program sorts the original list by dividing the numbers in the list into 1000 non-overlapping adjacent buckets. The `bucketsort` function finds the statistical range of the numbers in the original list (i.e. R = max(N) - min(N)) to help map each number into the appropriate bucket. The `insertion_sort` subroutine is then used to sort the numbers in each of these buckets. The sorted adjacent buckets are then appended onto the output list S which is the sorted version of the original list. This method is O(n²) in the worst case and O(n) on average. The large number of buckets (1000 buckets) was chosen to decrease the time needed to sort the final test case. If too few buckets are used, the time limit gets exceeded for the final test case. It is able to sort all test cases in 196 s.

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def insertion_sort(A):
            for i in range(1,len(A)):
                for j in range(0,i):
                    if A[i] < A[j]:
                        A.insert(j, A.pop(i))
                        break
            return A
        
        def bucketsort(A):
            buckets, m, S = [[] for _ in range(1000)], min(A), []
            R = max(A) - m
            if R == 0: return A
            for a in A: 
                buckets[999*(a-m)//R].append(a)
            for b in buckets: S.extend(insertion_sort(b))
            return S
    
        return bucketsort(nums)
    #https://leetcode.com/problems/sort-an-array/discuss/461394/Python-3-(Eight-Sorting-Algorithms)-(With-Explanation)
```

#### 88. Merge Sorted Array

#### 1122. Relative Sorted Array

#### 1086. High Five

#### 56. Merge Intervals

#### 75. Sort Colors

#### 1366 Rank Teams by Votes

#### Meeting Rooms two

#### 973 K Closest Points to Origin

#### 1288 Remove Covered Intervals

### Recursion

Tree, dfs, backtrack

### Stack

### Queue

Bfs, priortyqueue, compare function

### Linkedlist

#### 



### hashmap

#### 3. Longest Substring Without Repeating Characters

```python
class Solution:
    """
        :type s: str
        :rtype: int
    """
    def lengthOfLongestSubstring(self, s):
        start = 0 # pointer
        maxLength = 0 # answer
        usedChar = {} # a dictionary to store the used char, key : used char, value: the lastest index the char occurs
        
        for index,char in enumerate(s):
            if char in usedChar and start <= usedChar[char]:# case when we have repeated char from start to current char
                start = usedChar[char] + 1 # make sure the start point do not include the used char 
            else:
                maxLength = max(maxLength, index - start + 1) # the index starts from 0
        
            usedChar[char] = index # update the dictionary
        return maxLength
```



#### 570



### graph 

### DP

### Design

design hashmap, LRU cache, Trie, Design



### Merge k Sorted Lists

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        h = []
        for head in lists:
            node = head
            while node:
                h.append(node.val)
                node = node.next
                
        if not h:
            return None
        heapq.heapify(h)
        
        root = ListNode(heapq.heappop(h))
        curnode = root
        
        while h:
            nextNode = ListNode(heapq.heappop(h))
            # 一个新的node
            curnode.next = nextNode
            
            curnode = nextNode
            # 或者是curnode = curnode.next
            # 为什么curnode.next和curnode都需要等于nextNode
            # 最后一个位置会重复两个相同的数字吗
            
        return root
```

## DFS 树和图

### 144. Binary Tree Preorder Traversal

[Python3] Pre, In, Post Iteratively Summarization
In preorder, the order should be



root -> left -> right



But when we use stack, the order should be reversed:



right -> left -> root



Pre



```
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res, stack = [], [(root, False)]
        while stack:
            node, visited = stack.pop()  # the last element
            if node:
                if visited:  
                    res.append(node.val)
                else:  # preorder: root -> left -> right
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                    stack.append((node, True))
        return res
```



### 94. Binary Tree Inorder Traversal

In inorder, the order should be
left -> root -> right



But when we use stack, the order should be reversed:



right -> root -> left



In



```
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res, stack = [], [(root, False)]
        while stack:
            node, visited = stack.pop()  # the last element
            if node:
                if visited:
                    res.append(node.val)
                else:  # inorder: left -> root -> right
                    stack.append((node.right, False))
                    stack.append((node, True))
                    stack.append((node.left, False))
        return res
```

```
> 类型：
> Time Complexity O(n)
> Space Complexity O(1)
```

#### DFS Recursive

```
class Solution(object):
    def inorderTraversal(self, root):
        self.res = []
        self.dfs(root)
        return self.res
        
    def dfs(self, root):
        if not root:
            return
        self.dfs(root.left)
        self.res.append(root.val)
        self.dfs(root.right)
```

#### Stack



先把迭代到最左边的叶子节点，把所有途中的`root`放进stack，当左边走不通了，开始往`res`里面存数，并往右边走。



```
class Solution(object):
    def inorderTraversal(self, root):
        res, stack = [], []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                res.append(node.val)
                root = node.right
        return res
```

### 145 Binary Tree Postorder Traversal

In postorder, the order should be
left -> right -> root



But when we use stack, the order should be reversed:



root -> right -> left



Post



```
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res, stack = [], [(root, False)]
        while stack:
            node, visited = stack.pop()  # the last element
            if node:
                if visited:
                    res.append(node.val)
                else:  # postorder: left -> right -> root
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
        return res
```



### 101. Symmetric Tree

Recursive

```
class Solution:
    def isMirror(self, t1,t2):
        if(t1 is None and t2 is None):
            return True
        if(t1 is None or t2 is None):
            return False
        return(t1.val == t2.val) and self.isMirror(t1.right,t2.left) and self.isMirror(t1.left,t2.right)

    def isSymmetric(self, root: TreeNode) -> bool:
        return self.isMirror(root,root)

        
```

```
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def _symmetric(n1, n2):        
            if not n1 and not n2:
                return True
            if (not n1 and n2) or (not n2 and n1):
                return False
            return n1.val == n2.val and _symmetric(n1.left, n2.right) and _symmetric(n1.right, n2.left)
           
        return _symmetric(root, root)
```



iterative

```
class Solution:
    def isSymmetric(self, root):
        stack = []
        if root: stack.append([root.left, root.right])

        while(len(stack) > 0):
            left, right = stack.pop()
            
            if left and right:
                if left.val != right.val: return False
                stack.append([left.left, right.right])
                stack.append([right.left, left.right])
        
            elif left or right: return False
        
        return True
```

### 226. Invert Binary Tree

```
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

BFS

```
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return root
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            node.left, node.right = node.right, node.left
        return root
```

#### 572. Subtree of another tree

#### 104. Maximum Depth of Binary Tree

#### 100. Same Tree

#### 993. Cousins in Binary Tress

#### 938. Range Sum in BST

#### 617. Merge Two Binary Trees 

#### 112. Path Sum

#### 113. Path Sum two

#### 111. Minimum Depth of Binary Tree

#### 236. Lowest Common Ancestor of a Binary Search Tree

#### 108. Convert Sorted Array to Binary Search Tree

#### 257. Binary Tree Paths

#### 404. Sum of Left Leaves

#### 543. Diameter of Binary Tree

#### Trim a Binary Search Tree





## Daily Challenge

### Number of 1 Bits

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            count += n % 2
            n = n >> 1
        return count
```



```python
class Solution:
    def hammingWeight(self, n):
        ans = 0
        while n:
            n &= (n-1)
            ans += 1
        return ans
```



