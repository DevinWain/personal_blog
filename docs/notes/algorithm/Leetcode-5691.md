---
# icon, title, prev, original, next, sticky, copyrightText, mediaLink
category: algorithm
tags: [Leetcode, 双指针]
author: Wain
time: 2021-2-28
---
# Leetcode-5691.通过最少操作次数使数组的和相等

## 题目描述
给你两个长度可能不等的整数数组 `nums1` 和 `nums2` 。两个数组中的所有值都在 1 到 6 之间（包含 1 和 6）。

每次操作中，你可以选择 任意 数组中的任意一个整数，将它变成 1 到 6 之间 任意 的值（包含 1 和 6）。

请你返回使 `nums1` 中所有数的和与 `nums2` 中所有数的和相等的最少操作次数。如果无法使两个数组的和相等，请返回 -1 。

> 来源：力扣（LeetCode）<br>
> 链接：https://leetcode-cn.com/problems/equal-sum-arrays-with-minimum-number-of-operations

## 样例

### 样例一

**输入：** nums1 = [1,2,3,4,5,6], nums2 = [1,1,2,2,2,2]

**输出：** 3

**解释：** 你可以通过 3 次操作使 nums1 中所有数的和与 nums2 中所有数的和相等。以下数组下标都从 0 开始。

- 将 nums2[0] 变为 6 。 nums1 = [1,2,3,4,5,6], nums2 = [6,1,2,2,2,2] 。
- 将 nums1[5] 变为 1 。 nums1 = [1,2,3,4,5,1], nums2 = [6,1,2,2,2,2] 。
- 将 nums1[2] 变为 2 。 nums1 = [1,2,2,4,5,1], nums2 = [6,1,2,2,2,2] 。


### 样例二

**输入：** nums1 = [1,1,1,1,1,1,1], nums2 = [6]

**输出：** -1

**解释：** 没有办法减少 nums1 的和或者增加 nums2 的和使二者相等。

## 自己的想法

​	看到这道题时其实是跳过了周赛的第二题，直接做第三题，感觉比上一题更有思路。不难发现，返回-1的情况是某个数组长度大于另一数组的6倍，这可以优先判断。其次，必须先计算每个数组的总和，同时计算差值，这个差值就决定了操作次数。另一个要点是判断哪个数组总和小，则这个数组要增大，另一个减小。（这里我处理得不好，创建了新变量来保存，实际上只需要`swap`函数，确保一个是大的，另一个是小的）。

​	接着很自然的想法是从增幅（减幅）大的开始调整（只要最大的能调整成功，中间的值肯定可以，有点贪心思想），如小数组1调到6，大数组6调到1等。考虑到小数组1调到6，大数组6调到1等都是使差值减5，我把所有情况分了类，分成减5、4、3、2、1这五种，依次。于是我就弄了个哈希表来记录这五种情况的数量（这里没处理好，两个数组弄了两个表，实际可以合成一个）。有了这个表就能从5开始往下迭代累加，直到累加值刚好大于等于差值就说明可以调整到两者相等，输出对应的步数。（这里我是直接用差值5（4、3、...）乘以其个数来估计累加值，后面超过了再**回溯**一下，感觉可以节省一些迭代次数，但是会写得更长一点）

​	代码如下（有很多地方可以优化，一开始确实不太熟悉）：

```cpp
class Solution {
public:
    int minOperations(vector<int>& nums1, vector<int>& nums2) {
        int sum1=0, sum2=0, sub=0, add=0, count=0, tmp=0, value=0;
         vector<int> large, small;
        // 无法操作情况
        if(nums1.size()>6*nums2.size()||nums2.size()>6*nums1.size())
            return -1;
        // 计算总和，可以用stl的accumulate
        for(int i=0; i<nums1.size(); i++)
            sum1 += nums1[i];
        for(int i=0; i<nums2.size(); i++)
            sum2 += nums2[i];
        if(sum1 == sum2)
            return 0;
        // 设定大的数组，可以用stl的swap
        else if(sum1>sum2)
        {
            large = nums1;
            small = nums2;
            sub = sum1-sum2;
        }   
        else
        {
            small = nums1;
            large = nums2;
            sub = sum2-sum1;
        }
        // map可以用vector数组替换，两个数组可以合并
        map<int, int> s, l;
        for(int i=0; i<small.size(); i++)
        {
            s[small[i]] += 1;
        }
        for(int i=0; i<large.size(); i++)
        {
            l[large[i]] += 1;
        }
        for(int i=1; i<6; i++)
        {
            tmp = (s[i]+l[7-i]);
            value = (6-i);
            add += tmp*value;
            count += tmp;
            if(add<sub)
                continue;
            else if (add == sub)
                break;
            // 回溯
            else
            {
                for(int j=1; j<tmp; j++)
                {
                    add -= value;
                    count --;
                    // 注意相等跟小于的情况有点区别
                    if(add == sub)
                        return count;
                    else if(add<sub)
                    {
                        count ++;
                        return count;
                    }        
                }
            }
        }
        return count;
        
    }
};

```

## 题解思路

​	很多大佬陆续发布了题解，有个跟我思路比较像的，他优化得更好，直接看成对`nums1`进行调整，并记录两个数组，`inc`与`dec`，分别对应`nums1`需要增大来追上`nums2`或者减小。`inc`与`dec`在两个数组的累加时就可以开始哈希，若`nums1`值为6，`inc`不记录（不能增加了），`dec`在索引5处加一（可以减一个5），对于`nums2`，如果值为6，`inc`在索引5处加一（6变1时可以使`nums1`相对来说加5），`dec`不记录（不能使`nums1`相对减小）。有了这两个数组，只要`nums1`总和大于`nums2`时对`dec`迭代，判断临界时的步数即可，小于时的情况类似，等于时直接返回0。

​	以下是用户`lucifer1004`的代码，[题解原文](https://leetcode-cn.com/problems/equal-sum-arrays-with-minimum-number-of-operations/solution/tan-xin-ji-shu-pai-xu-by-lucifer1004-oa0l/):

```cpp
class Solution {
public:
    int minOperations(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), m = nums2.size();
        if (n > m * 6 || m > n * 6)
            return -1;
        int s1 = 0, s2 = 0;
        vector<int> inc(6), dec(6);
        for (int num : nums1) {
            s1 += num;
            if (num < 6)
                inc[6 - num]++;
            if (num > 1)
                dec[num - 1]++;
        }
        for (int num : nums2) {
            s2 += num;
            if (num < 6)
                dec[6 - num]++;
            if (num > 1)
                inc[num - 1]++;
        }
        if (s1 == s2)
            return 0;
        
        int cnt = 0;
        if (s1 > s2) {
            for (int i = 5; i >= 1; --i) {
                while (dec[i]) {
                    s1 -= i;
                    dec[i]--;
                    cnt++;
                    if (s1 <= s2)
                        return cnt;
                }
            }
        } else {
            for (int i = 5; i >= 1; --i) {
                while (inc[i]) {
                    s1 += i;
                    inc[i]--;
                    cnt++;
                    if (s1 >= s2)
                        return cnt;
                }
            }
        }
        
        return -1;
    }
};
```

​	另一种思路是用双指针，但必须先对两个数组进行排序，大数组从大开始迭代，小数组从小开始迭代，直到达到临界情况或迭代完数组。(类似于合并两个排序数组)

​	这里贴出`rainn`的代码，[题解原文](https://leetcode-cn.com/problems/equal-sum-arrays-with-minimum-number-of-operations/solution/c-shuang-zhi-zhen-tan-xin-by-rainn-dsdq/):

```cpp
class Solution {
public:
    int minOperations(vector<int>& a, vector<int>& b) {
        sort(a.begin(), a.end());
        sort(b.begin(), b.end());
        int sa = accumulate(a.begin(), a.end(), 0);
        int sb = accumulate(b.begin(), b.end(), 0);
        if (sa > sb) swap(sa, sb), swap(a, b);
        int i = 0, j = b.size() - 1;
        int cnt = 0;
        while (i < a.size() && 0 <= j && sa < sb) {
            if (6 - a[i] > b[j] - 1) { // 选择变化差值最大的一边
                sa += 6 - a[i++];
            } else sb -= b[j--] - 1;
            ++cnt;
        }
        while (i < a.size() && sa < sb) {
            sa += 6 - a[i++];
            ++cnt;
        }
        while (0 <= j && sa < sb) {
            sb -= b[j--] - 1;
            ++cnt;
        }
        return sa >= sb ? cnt : -1;
    }
};
```

## 总结

- 可以合理利用`stl`函数，如`accumulate(\#include<numeric>)`,`swap`等。其中`accumulate`带有三个形参：头两个形参指定要累加的元素范围，第三个形参则是累加的初值。
- `if`记得加`{}`
- 简单哈希不一定用`map`，可以用`vector`