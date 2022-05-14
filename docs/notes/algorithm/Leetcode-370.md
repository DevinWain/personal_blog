---
# icon, title, prev, original, next, sticky, copyrightText, mediaLink
category: algorithm
tags: [Leetcode, 数组]
author: Wain
time: 2022-5-14
---

# Leetcode-370.区间加法——差分数组java

## 题目描述

假设你有一个长度为 `n` 的数组，初始情况下所有的数字均为 `0`，你将会被给出 `k` 个更新的操作。

其中，每个操作会被表示为一个三元组：`[startIndex, endIndex, inc]`，你需要将子数组 `A[startIndex ... endIndex]`（包括 `startIndex` 和 `endIndex`）增加 `inc`。

请你返回 `k` 次操作后的数组。
[Leetcode37--区间加法](https://leetcode.cn/problems/range-addition)

## 样例

```java
输入: length = 5, updates = [[1,3,2],[2,4,3],[0,2,-2]]
输出: [-2,0,3,5,3]
```

- 初始状态:
  [0,0,0,0,0]

- 进行了操作 [1,3,2] 后的状态:
  [0,2,2,2,0]

- 进行了操作 [2,4,3] 后的状态:
  [0,2,5,5,3]

- 进行了操作 [0,2,-2] 后的状态:
  [-2,0,3,5,3]

## 解题思路

刚学了差分数组，来做这个题试试。流程是：构建差分数组、记录边界变化、还原数组。按照流程来其实不难，而且这个题比较特殊，原数组全是0，所以连res数组都不需要了，直接在diff数组上也能还原出来，还原语句如下：

```java
for(int i = 1; i < length; i++) {
    diff[i] += diff[i - 1];
  }
```

### 代码

```java
class Solution {
  public int[] getModifiedArray(int length, int[][] updates) {
    int[] diff = new int[length];
    int[] res = new int[length];
    for(int i = 0; i < updates.length; i++){
      diff[updates[i][0]] += updates[i][2];
      if(updates[i][1] + 1 < length){
        diff[updates[i][1] + 1] -= updates[i][2];
      }
    }
    res[0] = diff[0];
    for(int i = 1; i < length; i++){
      res[i] = res[i-1] + diff[i];
    }
    return res;
  }
}
```

