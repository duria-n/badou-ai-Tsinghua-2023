# # # import numpy as np
# # #
# # # # 示例中的真实框（ground truth boxes）
# # # boxes = np.array([
# # #     [100,300,400,150],  # [xmin, ymin, xmax, ymax]
# # #     [100, 150, 400, 300],
# # # ])
# # #
# # # # 示例中的先验框（prior boxes / anchor boxes）
# # # priors = np.array([
# # #     [200, 200, 500, 50],
# # #     [200, 50, 500, 200],
# # # ])
# # #
# # # # 计算交集区域的左上角坐标
# # # inter_upleft = np.maximum(priors[:, :2], boxes[:, :2])
# # # # 计算交集区域的右下角坐标
# # # inter_botright = np.minimum(priors[:, 2:], boxes[:, 2:])
# # #
# # # # 计算交集区域的宽度和高度
# # # inter_wh = np.maximum(inter_botright - inter_upleft, 0)
# # # # 计算交集区域的面积
# # # inter_area = inter_wh[:, 0] * inter_wh[:, 1]
# # #
# # # print("交集区域的左上角坐标：\n", inter_upleft)
# # # print("交集区域的右下角坐标：\n", inter_botright)
# # # print("交集区域的宽度和高度：\n", inter_wh)
# # # print("交集区域的面积：\n", inter_area)
# # # import numpy as np
# # #
# # # a = np.array([1, 2, 3])  # 原始数组
# # # b = np.tile(a, (2,1))  # 沿第一个维度复制2次
# # # c = np.tile(a, (2, 3))  # 沿第一个维度复制2次，第二个维度复制3次
# # #
# # # print(b)
# # # # 输出: [1 2 3 1 2 3]
# # #
# # # print(c)
# # # # 输出:
# # # # [[1 2 3 1 2 3 1 2 3]
# # # # #  [1 2 3 1 2 3 1 2 3]]
# # # import numpy as np
# # # from config import Config
# # #
# # # config = Config()
# # # size = config.anchor_box_scale
# # # ratios = config.anchor_box_ratios
# # # num_anchors = len(size)*(len(ratios))
# # # anchors = np.zeros((num_anchors,4))
# # # print(anchors)
# # #
# # # print(anchors)
# # # anchors[:,2:] = np.tile(size,(2,len(ratios))).T
# # #
# # #
# # # print(size)
# # # print(num_anchors)
# # # print(anchors)
# # class Solution(object):
# #     def copyRandomList(self, head):
# #         if head is None:
# #             return None
# #
# #         curr = head
# #         while curr:
# #             tmp = Node(curr.val)
# #             tmp.next = curr.next
# #             curr.next = tmp
# #             curr = tmp.next
# #
# #         pre = head
# #         res = head.next
# #         curr = head.next
# #         res_pre = head
# #         while curr:
# #             if curr.random:
# #                 curr.random = pre.random.next
# #             res_pre = curr
# #             pre = curr.next
# #             if pre:
# #                 curr = pre.next
# #             res_pre.next = curr
# #
# #         return res
# #
# class MinStack(object):
#
#     def __init__(self):
#         """
#         initialize your data structure here.
#         """
#         self.A, self.B = [], []
#
#     def push(self, x):
#         """
#         :type x: int
#         :rtype: None
#         """
#         self.A.append(x)
#         if not self.B or self.B[-1] >= x:
#             self.B.append(x)
#
#     def pop(self):
#         """
#         :rtype: None
#         """
#         if self.A.pop() == self.B[-1]:
#             self.B.pop()
#
#     def top(self):
#         """
#         :rtype: int
#         """
#         return self.A[-1]
#
#     def min(self):
#         """
#         :rtype: int
#         """
#         return self.B[-1]
#
#
# # Your MinStack object will be instantiated and called as such:
class ListNode(object):
    def __init__(self,x):
        self.val= x
        self.next = None

    def gettree(self,head,arr):
        pre = head
        for i in arr:
            tmp = ListNode(i)
            pre.next = tmp
            pre = tmp


def recur(cur, pre):
    if not cur: return pre  # 终止条件
    res = recur(cur.next, cur)  # 递归后继节点
    cur.next = pre  # 修改节点引用指向
    return res  # 返回反转链表的头节点

headarr = [2,3,4,5,6]
head = ListNode(1)
head.gettree(head,headarr)

res= recur(head, None)  # 调用递归并返回
print(res)