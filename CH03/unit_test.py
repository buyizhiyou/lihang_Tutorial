# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: unit_test
# Date: 8/15/18
# Author: 😏 <smirk dot cao at gmail dot com>
from knn import *
import numpy as np
import argparse
import logging
import unittest


class TestStringMethods(unittest.TestCase):
    def test_q32(self):
        data = np.loadtxt("Input/data_3-2.txt")
        X,Y = data[:,:2],data[:,2]
        target = np.array([2, 4.5])
        clf = KNNKdTree()
        clf.fit(X,Y)
        nearest,label = clf.predict(target)
        self.assertEqual(1,label)
        logger.info(label)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    # unittest.main()
    # 构造测试集：第一步，创建一个测试套件,TestSuite用来装一个或多个测试用例(多个测试方法)
    suite = unittest.TestSuite()
    # 添加单条测试方法：只有被添加的测试方法才会被执行
    suite.addTest(TestStringMethods("test_q32"))
    # suite.addTest(TestStringMethods("test_e31"))
    # 执行测试
    runner = unittest.TextTestRunner()
    runner.run(suite)

