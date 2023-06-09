import cv2
import numpy as np
 
 
class SealRemove(object):
 
    def remove_red_seal(self, image):
        # 获得红色通道
        blue_c, green_c, red_c = cv2.split(image)
        # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
        thresh, ret = cv2.threshold(red_c, 210, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 实测调整为95%效果好一些
        filter_condition = int(thresh * 0.99)
 
        nominator_thresh, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)
        return red_thresh
 
 
    def shijuanqingli(self, image):
        # img = cv2.imread(image, 0)
        thresh, dst1 = cv2.threshold(image,210, 255, cv2.THRESH_BINARY)
        dst1_without_pen = dst1
        return dst1_without_pen
 
    def join_image(self, img_without_red, dst1_without_pen):
        ret = cv2.bitwise_or(img_without_red, dst1_without_pen)
        return ret
 
if __name__ == '__main__':
    src = r'/data/wwwroot/default/asset/testsign2.png'
    image0 = cv2.imread(src)
    seal_rm = SealRemove()
    image_0 = seal_rm.remove_red_seal(image0)
    # image_0_1 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    print("<img src='/asset/testsign2.png' />")
    image1 = cv2.imread(src, 0)
    image_1 = seal_rm.shijuanqingli(image1)
    image_result = seal_rm.join_image(image_0, image_1)
 
   # cv2.imshow('new image', image_result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
 
    #src_temp = src.split(r'.')
    src_result = '/data/wwwroot/default/Data/newsign.png'
 
    cv2.imwrite(src_result,image_result)
    print("<img src='/Data/newsign.png' />")
		