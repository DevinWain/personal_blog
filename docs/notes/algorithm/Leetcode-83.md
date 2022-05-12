---
# icon, title, prev, original, next, sticky, copyrightText, mediaLink
category: algorithm
tags: [Leetcode, 双指针]
author: Wain
time: 2022-5-12
---
# Leetcode-83.删除链表重复元素——双指针cpp



## 题目描述

给定一个已排序的链表的头`head`，删除所有重复的元素，使每个元素只出现一次。返回已排序的链表。

[题目链接——83.删除链表重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)

```
输入：head = [1,1,2]
输出：[1,2]
```

```
输入：head = [1,1,2,3,3]
输出：[1,2,3]
```

**提示**：

- 链表中节点数目在范围 `[0, 300]` 内
- `-100 <= Node.val <= 100`
- 题目数据保证链表已经按升序 **排列**

## 解题思路



这一题依然是链表删除元素类型的题目，也仍然能用双指针来做。思路也很简单，一个指针向前探路，遇到跟慢指针不同的值就接过去，最后慢指针连接一下`NULL`就行。



这题还有一个点需要注意，删除节点后，这些节点按理会悬空，没有真正释放。对于有垃圾自动回收的语言，那还可以不管。但是像cpp这种，可能还得释放一下。我找了一下评论，释放的代码如下：



```cpp
if (pi->val == pi->next->val) {
    ListNode *del = pi->next;
    pi->next = pi->next->next;
    delete del;
}
```



但是delete也不是随便能用的，只能释放堆内存对象，对栈内存对象不管用。不过很多时候都是处理堆内存对象，包括本题也是能用的。



## 代码



```cpp
/**

 \* Definition for singly-linked list.
 \* struct ListNode {
 \*   int val;
 \*   ListNode *next;
 \*   ListNode() : val(0), next(nullptr) {}
 \*   ListNode(int x) : val(x), next(nullptr) {}
 \*   ListNode(int x, ListNode *next) : val(x), next(next) {}
 \* };
 */

class Solution {

public:

  ListNode* deleteDuplicates(ListNode* head) {
    if(!head)
      return NULL;
    ListNode* first = head;
    ListNode* last = head;
    
    while(true){
      first = first->next;      
      if(!first){
        last->next = NULL;
        break; 
      }    
      if(first->val != last->val){
        last->next = first;
        last = last->next;
      }
    }
    return head;
  }
};
```