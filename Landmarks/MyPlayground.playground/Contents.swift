import UIKit

public class ListNode {
    public var val: Int
    public var next: ListNode?
    public init() {
        self.val = 0;
        self.next = nil;
    }
    
    public init(_ val: Int) {
        self.val = val
        self.next = nil
    }
    
    public init(_ val: Int, _ next: ListNode?) {
        self.val = val;
        self.next = next;
    }
}

/**
 *  9 回文数
 */
func isPalindrome(_ x: Int) -> Bool {
    var temp = x
    if temp < 0 || temp % 10 == 0, temp != 0 {
        return false
    }
    var revertNum = 0
    while (temp > revertNum) {
        revertNum = revertNum * 10 + temp % 10
        temp =  temp / 10
    }
    return temp == revertNum || temp == revertNum / 10
}

/**
 * 24 反转链表
 * Definition for singly-linked list.
 */

func reverseList(_ head: ListNode?) -> ListNode? {
    guard let tHead = head, let next = tHead.next else { return head }
    let newHead = reverseList(next)
    tHead.next?.next = tHead
    tHead.next = nil
    return newHead
}

/**
 * 509 斐波那契额数列
 */
func fib(_ n: Int) -> Int {
    if n < 2 { return n }
    var p = 0, q = 0, r = 1
    for _ in 2...n {
        p = q
        q = r
        r = p + q
    }
    return r
}

/**
 * 11 盛水最多
 * time: O(n)
 * space: O(1)
 */
func maxArea1(_ height: [Int]) -> Int {
    var maxA = 0, left = 0, right = height.count - 1
    while left < right {
        maxA = max(maxA, (right - left) * min(height[left], height[right]))
        if height[left] < height[right] {
            left += 1
        } else if height[left] > height[right] {
            right -= 1
        }
    }
    return maxA
}

func maxArea2(_ height: [Int]) -> Int {
    var maxA = 0, currentA = 0, left = 0, leftLength = 0, right = height.count - 1, rightLength = 0
    
    while left < right {
        if height[left] <= 0 {
            left += 1
            continue
        }
        
        if height[right] <= 0 {
            right -= 1
            continue
        }
        
        if height[left] < leftLength {
            leftLength = height[left]
            left += 1
            continue
        }
        
        if height[right] < rightLength {
            rightLength = height[right]
            right -= 1
            continue
        }
        
        leftLength = height[left]
        rightLength = height[right]
        currentA = min(leftLength, rightLength) * (right - left)
        maxA = max(maxA, currentA)
        
        if leftLength < rightLength {
            left += 1
        } else {
            right -= 1
        }
    }
    return maxA
}

/**
 * 344 反转字符串
 * time: O(n)
 * space: O(1)
 */
func reverseString(_ s: inout [Character]) {
    var lIndex = 0, rIndex = s.count - 1
    
    while lIndex < rIndex {
        let temp = s[rIndex]
        s[rIndex] = s[lIndex]
        s[lIndex] = temp
        
        lIndex += 1
        rIndex -= 1
    }
    
}

/**
 *  2 两数相加
 */
func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    var l1 = l1, l2 = l2
    var needAdd = 0
    
    let newHead = ListNode.init(-1)
    var cur = newHead
    
    while l1 != nil || l2 != nil {
        let new = (l1?.val ?? 0) + (l2?.val ?? 0) + needAdd
        needAdd = new / 10
        cur.next = ListNode.init(new % 10)
        
        cur = cur.next!
        l1 = l1?.next
        l2 = l2?.next
    }
    
    if needAdd == 1 {
        cur.next = ListNode.init(1)
    }
    
    return newHead.next
}

/**
 *  383 赎金信
 *  给你两个字符串：ransomNote 和 magazine ，判断 ransomNote 能不能由 magazine 里面的字符构成。
 *  如果可以，返回 true ；否则返回 false 。
 *  magazine 中的每个字符只能在 ransomNote 中使用一次。
 *  链接：https://leetcode.cn/problems/ransom-note
 *  解答：https://leetcode.cn/problems/ransom-note/solution/dai-ma-sui-xiang-lu-dai-ni-gao-ding-ha-x-5pak/
 *  time O(n)
 *  space O(1)
 */

func canConstruct(_ ransomNote: String, _ magazine: String) -> Bool {
    if ransomNote.count == 0 { return true }
    print(ransomNote.count, magazine.count);
    
    guard ransomNote.count <= magazine.count else { return false }
    
    var record = Array(repeating: 0, count: 26)
    let aUnicodeScalerValue = "a".unicodeScalars.first!.value
    for unicodeScaler in magazine.unicodeScalars {
        let idx: Int = Int(unicodeScaler.value - aUnicodeScalerValue)
        record[idx] += 1
    }
    
    for unicodeScaler in ransomNote.unicodeScalars {
        let idx: Int = Int(unicodeScaler.value - aUnicodeScalerValue)
        record[idx] -= 1
        if record[idx] < 0 {
            return false
        }
    }
    return true
}


/**
 *  454 四数相加 II
 *  给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：
 *  0 <= i, j, k, l < n
 *  nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
 *  链接：https://leetcode.cn/problems/4sum-ii
 *  解答：https://leetcode.cn/problems/4sum-ii/solution/454-si-shu-xiang-jia-iimapzai-ha-xi-fa-zhong-de-2/
 *  time O(n^2)
 *  spaceO(n^2)
 */
func fourSumCount(_ nums1: [Int], _ nums2: [Int], _ nums3: [Int], _ nums4: [Int]) -> Int {
    var map = [Int: Int]()
    
    for i in 0..<nums1.count {
        for j in 0..<nums2.count {
            let sum1 = nums1[i] + nums2[j]
            map[sum1] = (map[sum1] ?? 0) + 1
        }
    }
    
    var res = 0
    for i in 0..<nums3.count {
        for j in 0..<nums4.count {
            let sum2 = nums3[i] + nums4[j]
            let other = 0 - sum2
            if map.keys.contains(other) {
                res += (map[other] ?? 0)
            }
        }
    }
    return res
    
}
//let res = fourSumCount([-1, -1], [-1, 1], [-1, 1], [1, -1])


/**
 *  202 快乐数
 *  编写一个算法来判断一个数 n 是不是快乐数。
 *  「快乐数」 定义为：
 *  对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
 *  然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
 *  如果这个过程 结果为 1，那么这个数就是快乐数。
 *  如果 n 是 快乐数 就返回 true ；不是，则返回 false 。
 *  链接：https://leetcode.cn/problems/happy-number
 *  解答：https://leetcode.cn/problems/happy-number/solution/dai-ma-sui-xiang-lu-dai-ni-gao-ding-ha-x-sx0j/
 *  time O()
 *  spaceO()
 */

func getSum(_ number: Int) -> Int {
    var sum = 0, num = number
    while num > 0 {
        let temp = num % 10
        sum += (temp * temp)
        num /= 10
    }
    return sum
}

func isHappy(_ n: Int) -> Bool {
    var set = Set<Int>()
    var num = n
    while true {
        let sum = getSum(num)
        if sum == 1 {
            return true
        }
        
        if set.contains(sum) {
            return false
        } else {
            set.insert(sum)
        }
        num = sum
    }
}

/**
 *  7 整数反转
 *  给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
 *  如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。
 *  假设环境不允许存储 64 位整数（有符号或无符号）。
 *  链接：https://leetcode.cn/problems/reverse-integer
 *  time O()
 *  space O()
 */

func reverse(_ x: Int) -> Int {
    var res = 0
    var x = x
    while x != 0 {
        res = res * 10 + x % 10
        if res > Int32.max || res < Int32.min {
            return 0
        }
        x /= 10
    }
    return res
}


/**
 *  141 环形链表
 *  给你一个链表的头节点 head ，判断链表中是否有环。
 *  如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。
 *  如果链表中存在环 ，则返回 true 。 否则，返回 false 。
 *  链接：https://leetcode.cn/problems/linked-list-cycle
 *  time O()
 *  space O()
 */

func hasCycle(_ head: ListNode?) -> Bool {
    var slow = head
    var fast = head?.next
    
    while slow != nil, fast != nil {
        if slow === fast {
            return true
        }
        slow = slow?.next
        fast = fast?.next?.next
    }
    return false
}

/**
 * 142 环形链表 II
 * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
 * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
 * 不允许修改 链表。
 * 链接：https://leetcode.cn/problems/linked-list-cycle-ii
 *  time O()
 *  space O()
 */

func detectCycle(_ head: ListNode?) -> ListNode? {
    var slow = head
    var fast = head
    
    while fast != nil, fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
        
        if slow == fast {
            var list1 = slow
            var list2 = head
            while list1 != list2 {
                list1 = list1?.next
                list2 = list2?.next
            }
            return list2
        }
    }
    return nil
}

extension ListNode: Equatable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(val)
        hasher.combine(ObjectIdentifier(self))
    }
    
    public static func == (lhs: ListNode, rhs: ListNode) -> Bool {
        return lhs === rhs
    }
}


/**
 * 349 两个数组的交集
 * 给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。
 * 链接：https://leetcode.cn/problems/intersection-of-two-arrays/
 * 解答：https://leetcode.cn/problems/intersection-of-two-arrays/solution/liang-ge-shu-zu-de-jiao-ji-zhe-dao-ti-kao-cha-liao/
 * time O()
 * space O()
 */

func intersection(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
    var set1 = Set<Int>()
    var set2 = Set<Int>()
    
    for i in nums1 {
        set1.insert(i)
    }
    
    for i in nums2 {
        if set1.contains(i) {
            set2.insert(i)
        }
    }
    return Array.init(set2)
}

/**
 *  242 有效的字母异位词
 *  给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
 *  注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。
 *  链接：https://leetcode.cn/problems/valid-anagram
 *  解答：https://leetcode.cn/problems/valid-anagram/solution/by-carlsun-2-cph7/
 *  time O(n)
 *  space O(1)
 */
func isAnagram(_ s: String, _ t: String) -> Bool {
    if s.count != t.count { return false }
    
    var record = Array.init(repeating: 0, count: 26)
    let aUnicodeScalar = "a".unicodeScalars.first!.value
    
    for c in s.unicodeScalars {
        let idx = Int(c.value - aUnicodeScalar)
        record[idx] += 1
    }
    
    for c in t.unicodeScalars {
        let idx = Int(c.value - aUnicodeScalar)
        record[idx] -= 1
    }
    
    for value in record {
        if value != 0 {
            return false
        }
    }
    return true
}

/**
 *  203 移除链表元素
 *  给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
 *  链接：https://leetcode.cn/problems/remove-linked-list-elements/
 *  解答：https://leetcode.cn/problems/remove-linked-list-elements/solution/by-carlsun-2-e4wo/
 *  time O(n)
 *  space O(1)
 */
func removeElements(_ head: ListNode?, _ val: Int) -> ListNode? {
    let dummyNode = ListNode()
    dummyNode.next = head
    var currentNode = dummyNode
    while let curNext = currentNode.next {
        if curNext.val == val {
            currentNode.next = curNext.next
        } else {
            currentNode = curNext
        }
    }
    return dummyNode.next
}


/**
 * 977 有序数组的平方
 * 给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
 * 链接：https://leetcode.cn/problems/squares-of-a-sorted-array/
 * 解答：https://leetcode.cn/problems/squares-of-a-sorted-array/solution/dai-ma-sui-xiang-lu-shu-zu-ti-mu-zong-ji-1rtz/
 * time O(n)
 * space O(1)
 */

func sortedSquares(_ nums: [Int]) -> [Int] {
    var k = nums.count - 1
    var i = 0
    var j = nums.count - 1
    
    var result = Array.init(repeating: -1, count: nums.count)
    for _ in 0..<nums.count {
        if nums[i] * nums[i] < nums[j] * nums[j] {
            result[k] = nums[j] * nums[j]
            j -= 1
        } else {
            result[k] = nums[i] * nums[i]
            i += 1
        }
        k -= 1
    }
    return  result
}

/**
 * 27 移除元素
 * 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
 * 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
 * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
 * 链接：https://leetcode.cn/problems/remove-element
 * 解答：https://leetcode.cn/problems/remove-element/solution/by-carlsun-2-fdc4/
 * time O(n)
 * space O(1)
 */

func removeElement(_ nums: inout [Int], _ val: Int) -> Int {
    var slowIndex = 0
    for fastIndex in 0..<nums.count {
        if val != nums[fastIndex] {
            nums[slowIndex] = nums[fastIndex]
            slowIndex += 1
        }
    }
    return slowIndex
}


/**
 * 704 二分查找
 * 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
 * 链接：https://leetcode.cn/problems/binary-search
 * 解答：https://leetcode.cn/problems/binary-search/solution/dai-ma-sui-xiang-lu-dai-ni-xue-tou-er-fe-ajox/
 * time O()
 * space O()
 */

func search(_ nums: [Int], _ target: Int) -> Int {
    var leftIndex = 0, rightIndex = nums.count - 1
    
    while leftIndex <= rightIndex {
        let midIndex = leftIndex + (rightIndex - leftIndex) / 2
        if nums[midIndex] > target {
            rightIndex = midIndex - 1
            
        } else if nums[midIndex] < target {
            leftIndex = midIndex + 1
        } else {
            return midIndex
        }
    }
    return -1
}

/**
 * 1 两数之和
 * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
 * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
 * 你可以按任意顺序返回答案。
 * 链接：https://leetcode.cn/problems/two-sum
 * 解答：https://leetcode.cn/problems/two-sum/solution/by-carlsun-2-sarb/
 * time O(n)
 * space O(1)
 */

func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
    var dict = [Int: Int]()
    for (index, value) in nums.enumerated() {
        if let y = dict[target - value] {
            return [y, index]
        } else {
            dict[value] = index
        }
    }
    return []
}

/**
 * 6 Z字形变换
 * 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
 * 比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
 * P   A   H   N
 * A P L S I I G
 * Y   I   R
 * 之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
 * 请你实现这个将字符串进行指定行数变换的函数：
 * string convert(string s, int numRows);
 * 链接：https://leetcode.cn/problems/zigzag-conversion
 * 解答：
 */

func convert(_ s: String, _ numRows: Int) -> String {
    ""
}

/**
 *  5 最长回文子串
 *  给你一个字符串 s，找到 s 中最长的回文子串。
 *  如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
 *  链接：https://leetcode.cn/problems/longest-palindromic-substring/
 *  解答：
 *  time O()
 *  space O()
 */
func longestPalindrome(_ s: String) -> String {
    ""
}

/**
 * 3 无重复字符的最长子串
 * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
 * 链接：https://leetcode.cn/problems/longest-substring-without-repeating-characters/
 * 解答：
 * time O()
 * space O()
 */

func lengthOfLongestSubstring(_ s: String) -> Int {
    0
}

/**
 *  剑指Offer 29. 顺时针打印矩阵
 *  入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
 *  输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
 *  输出：[1,2,3,6,9,8,7,4,5]
 *  链接：https://leetcode.cn/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/
 *  解答：
 *  time O()
 *  space O()
 */
func spiralOrder(_ matrix: [[Int]]) -> [Int] {
    []
}
