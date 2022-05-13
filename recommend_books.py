# -*-coding:utf-8-*-
import os
import django
import operator
from book.models import *
from math import sqrt, pow

os.environ["DJANGO_SETTINGS_MODULE"] = "book.settings"
django.setup()


class UserCf:
    # 基于用户协同算法来获取推荐列表
    """
    利用用户的群体行为来计算用户的相关性。
    计算用户相关性的时候我们就是通过对比他们对相同物品打分的相关度来计算的

    举例：

    --------+--------+--------+--------+--------+
            |   X    |    Y   |    Z   |    R   |
    --------+--------+--------+--------+--------+
        a   |   5    |    4   |    1   |    5   |
    --------+--------+--------+--------+--------+
        b   |   4    |    3   |    1   |    ?   |
    --------+--------+--------+--------+--------+
        c   |   2    |    2   |    5   |    1   |
    --------+--------+--------+--------+--------+

    a用户给X物品打了5分，给Y打了4分，给Z打了1分
    b用户给X物品打了4分，给Y打了3分，给Z打了1分
    c用户给X物品打了2分，给Y打了2分，给Z打了5分

    那么很容易看到a用户和b用户非常相似，但是b用户没有看过R物品，
    那么我们就可以把和b用户很相似的a用户打分很高的R物品推荐给b用户，
    这就是基于用户的协同过滤。
    """

    # 获得初始化数据
    def __init__(self, data):
        self.data = data

    # 通过用户名获得书籍列表，仅调试使用
    def getItems(self, username1, username2):
        return self.data[username1], self.data[username2]

    # 计算两个用户的皮尔逊相关系数
    def pearson(self, user1, user2):  # 数据格式为：书籍id，浏览次数
        print("user message", user1)
        sumXY = 0.0
        n = 0
        sumX = 0.0
        sumY = 0.0
        sumX2 = 0.0
        sumY2 = 0.0
        for movie1, score1 in user1.items():
            if movie1 in user2.keys():  # 计算公共的书籍浏览次数
                n += 1
                sumXY += score1 * user2[movie1]
                sumX += score1
                sumY += user2[movie1]
                sumX2 += pow(score1, 2)
                sumY2 += pow(user2[movie1], 2)
        if n == 0:
            print("p氏距离为0")
            return 0
        molecule = sumXY - (sumX * sumY) / n
        denominator = sqrt((sumX2 - pow(sumX, 2) / n) * (sumY2 - pow(sumY, 2) / n))
        if denominator == 0:
            print("共同特征为0")
            return 0
        r = molecule / denominator
        print("p氏距离:", r)
        return r

    # 计算与当前用户的距离，获得最临近的用户
    def nearest_user(self, username, n=1):
        distances = {}
        # 用户，相似度
        # 遍历整个数据集
        for user, rate_set in self.data.items():
            # 非当前的用户
            if user != username:
                distance = self.pearson(self.data[username], self.data[user])
                # 计算两个用户的相似度
                distances[user] = distance
        closest_distance = sorted(
            distances.items(), key=operator.itemgetter(1), reverse=True
        )
        # 最相似的N个用户
        print("closest user:", closest_distance[:n])
        return closest_distance[:n]

    # 给用户推荐书籍
    def recommend(self, username, n=1):
        recommend = {}
        nearest_user = self.nearest_user(username, n)
        for user, score in dict(nearest_user).items():  # 最相近的n个用户
            for book_id, scores in self.data[user].items():  # 推荐的用户的书籍列表
                if book_id not in self.data[username].keys():  # 当前username没有看过
                    rate = RateBook.objects.filter(book_id=book_id, user__username=user)
                    # 如果用户评分低于3分，则表明用户不喜欢此书籍，则不推荐给别的用户
                    if rate and rate.first().mark < 3:
                        continue
                    if book_id not in recommend.keys():  # 添加到推荐列表中
                        recommend[book_id] = scores
        # 对推荐的结果按照书籍浏览次数排序
        return sorted(recommend.items(), key=operator.itemgetter(1), reverse=True)


def recommend_by_user_id(user_id, book_id=None):
    # 通过用户协同算法来进行推荐
    current_user = User.objects.get(id=user_id)
    # 如果当前用户没有打分 则按照热度顺序返回
    if current_user.ratebook_set.count() == 0:
        if book_id:
            book_list = Book.objects.exclude(pk=book_id).order_by("-like_num")[:15]
        else:
            book_list = Book.objects.all().order_by("-like_num")[:15]
        return book_list
    users = User.objects.all()
    all_user = {}
    for user in users:
        rates = user.ratebook_set.all()
        rate = {}
        # 用户有给图书打分
        if rates:
            for i in rates:
                rate.setdefault(str(i.book.id), i.mark)
            all_user.setdefault(user.username, rate)
        else:
            # 用户没有为书籍打过分，设为0
            all_user.setdefault(user.username, {})

    print("this is all user:", all_user)
    user_cf = UserCf(data=all_user)
    recommend_list = user_cf.recommend(current_user.username, 15)
    good_list = [each[0] for each in recommend_list]
    print('this is the good list', good_list)
    if not good_list:
        # 如果没有找到相似用户喜欢的书则按照热度顺序返回
        if book_id:
            book_list = Book.objects.exclude(pk=book_id).order_by("-like_num")[:15]
        else:
            book_list = Book.objects.all().order_by("-like_num")[:15]
        return book_list
    if book_id and book_id in good_list:
        good_list.pop(good_list.index(book_id)) # 不推荐书籍book_id
    book_list = Book.objects.filter(id__in=good_list).order_by("-collect_num")[:15]
    return book_list


class ItemCf:
    # 基于物品协同算法来获取推荐列表
    '''
    本实现中物品相似度的计算公式为：
    wij = N(i)⋂N(j)/ sqrt(N(i)*N(j))
    其中N(i)表示书籍 i 被评分过的次数，N(i)⋂N(j)表示同时评分过书籍 i 和书籍 j 的用户数。
    具体步骤为：
    1、遍历数据集获取用户评分过的书籍列表A
    2、使用变量i遍历该用户评分过的书籍A，对评分过该书籍的用户数(即变量N(i)) + 1
    3、获取评分过书籍 i 的用户评分过的书籍列表B，使用变量 j 遍历这个列表，
        如果 i 和 j 不同，则 W[i][j]  + 1 ，此时W[i][j]记录的是同时评分过书籍 i 和书籍 j 的用户数）。
    4、遍历 W 矩阵，对 W 中的每一个元素进行如下计算：
        wij =  W[i][j] / sqrt(N(i)*N(j))
        得到的 W 矩阵就是物品之间的相似度矩阵。
    5、首先获得用户已经评分过的书籍列表，遍历该书籍列表，对于用户评分过的书籍 i ，找出与书籍 i 最相似的前 k 本书籍
        （对 W[i] 按照相似度排序），计算这 k 本书籍各自的加权评分(rank):
            rankj= sum(用户对书籍i的评分∗书籍i和书籍j的相似度)
    6、对rank按照评分倒序排序，取前 n 个推荐给用户即可。
    '''
    def __init__(self, user_id, book_id):
        self.book_id = book_id  # 书籍id
        self.user_id = user_id  # 用户id
        self.N = {}  # 用户互动过的物品数量
        self.W = {}  # 记录的是同时看过书籍 i 和书籍 j 的用户数的相似度矩阵

        self.train = {}

        # recommend n items from the k most similar to the items user have interacted
        self.k = 30 # 与书籍i最相似的前k本书籍
        self.n = 3 # 计算这 k 本书籍各自的加权评分，对rank按照评分倒序排序，取前 n 个推荐给用户即可。

    def get_rate_book(self):
        # 获取用户评分过的书籍
        rate_books = RateBook.objects.filter()
        if not rate_books:
            return False
        datas = {}
        for rate_book in rate_books:
            user_id = rate_book.user_id
            if user_id not in datas:
                datas[user_id] = [[rate_book.book.id, rate_book.mark]]
            else:
                datas[user_id].append([rate_book.book.id, rate_book.mark])

        return datas

    def get_user_rate_book(self, book_id):
        # 获取评分过书籍 i 的用户评分过的书籍列表B
        rate_books = RateBook.objects.filter(book_id=book_id)
        book_ids = []
        for rate_book in rate_books:
            for book in RateBook.objects.filter(user_id=rate_book.user_id):
                book_ids.append(book.book.id)
        return book_ids

    def similarity(self):
        """
        计算书籍i与书籍j的相似矩阵
        """

        self.train = self.get_rate_book()  # 获取用户评分过的书籍列表A
        if not self.train:
            # 用户没有评分过任何书籍
            return False

        for user, item_ratings in self.train.items():
            items = [x[0] for x in item_ratings]  # items that user have interacted
            for i in items:
                self.N.setdefault(i, 0)
                self.N[i] += 1  # number of users who have interacted item i
                for j in items:
                    if i != j:
                        self.W.setdefault(i, {})
                        self.W[i].setdefault(j, 0)
                        self.W[i][j] += 1  # number of users who have interacted item i and item j
        for i, j_cnt in self.W.items():
            for j, cnt in j_cnt.items():
                self.W[i][j] = self.W[i][j] / (self.N[i] * self.N[j]) ** 0.5

        return True

    def recommendation(self):
        """
        给用户推荐相似书籍
        """
        if not self.similarity():
            # 用户没有评分过任何书籍，就返回前3本热门书籍，按点赞量降序返回
            book_list = Book.objects.all().exclude(pk=self.book_id).order_by("-like_num")[:3]
            return book_list

        if self.user_id not in self.train:
            book_list = Book.objects.all().exclude(pk=self.book_id).order_by("-like_num")[:3]
            return book_list
        rank = {}
        watched_items = [x[0] for x in self.train[self.user_id]]
        for i in watched_items:
            for j, similarity in sorted(self.W[i].items(), key=operator.itemgetter(1), reverse=True)[0:self.k]:
                if j not in watched_items:
                    rank.setdefault(j, 0.)
                    rank[j] += float(self.train[self.user_id][watched_items.index(i)][
                                         1]) * similarity  # rating that user rate for item i * similarity between item i and item j
        sort_rank =  sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:self.n]
        print(sort_rank)
        if not sort_rank:
            # 用户没有评分过任何书籍，就返回前15本热门书籍，按点赞量降序返回
            book_list = Book.objects.all().exclude(pk=self.book_id).order_by("-like_num")[:3]
            return book_list
        book_list = Book.objects.filter(id__in=[s[0] for s in sort_rank]).exclude(pk=self.book_id).order_by("-like_num")[:3]
        return book_list
