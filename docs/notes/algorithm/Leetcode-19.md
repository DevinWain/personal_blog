---
# icon, title, prev, original, next, sticky, copyrightText, mediaLink
category: algorithm
tags: [Leetcode, 双指针]
author: Wain
time: 2022-5-11
---

# Leetcode-19 删除链表的倒数第`n`个结点

### 题目描述

​	给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。
   
   [题目链接](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/) 

### 样例

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

```
输入：head = [1], n = 1
输出：[]
```

```
输入：head = [1,2], n = 1
输出：[1]
```

**提示：**

- 链表中结点的数目为 sz
- 1 <= `sz`  <= 30
- 0 <= `Node.val`  <= 100
- 1 <= `n`  <= `sz` 

### 解题思路

​	感觉应该是第二次做这道题了，这次果断用双指针。虽说是双指针，实际上建了4个，但最有用的是那两个。同时要注意加个哨兵根节点，这样进行删除操作时更加方便，可以定位到前一个节点。核心是让first节点先从哨兵节点跑n+1步（head节点开始跑n步），然后再用一个循环定位到倒数n-1个节点，把倒数第n个删了，最后也只需要返回哨兵节点的下一个节点作为根节点。

### 代码

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* first = head;
        ListNode* dump = new ListNode(101);
        ListNode* last = dump;
        last->next = head;
        for(int i = 0; i < n; i++){
            first = first->next;
        }
        while(first){
            first = first->next;
            last = last->next;
        }
        ListNode* tmp = last;
        last = last->next;
        tmp->next = last->next;
        return dump->next;
    }
};

```

